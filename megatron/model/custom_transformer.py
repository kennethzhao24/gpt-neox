"""
    OPT Transformer.
"""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
 



class ParallelMLP(nn.Module):
    """
        MLP will take the input with h hidden state, project it to ffn
        hidden dimension, perform nonlinear transformation, and project the
        state back into h hidden dimension. At the end, dropout is also
        applied.
    """

    def __init__(
            self, 
            neox_args,
            init_method, 
            output_layer_init_method, 
            parallel_output=False
            ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        self.fc1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.ffn_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )

        self.fc2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.ffn_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.fc1(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.fc2(intermediate_parallel)
        return output, output_bias


class ParallelLinear(nn.Module):
    """
        A Parallel Linear Layer transforming the transformer outputs from hidden_size -> output_size
    """
    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=input_size,
                output_size=output_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if neox_args.use_mup = True, despite it not being included here
            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    """
        Parallel self-attention layer abstract class.

        Self-attention layer takes input with size [b, s, h]
        and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        use_cache=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        neox_args.num_attention_heads = neox_args.hidden_size // neox_args.head_dim

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=3 * neox_args.hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=neox_args.use_bias_in_attn_linear,
        )

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.attention_type = neox_args.attention_type
        self.use_flash_attention = self.attention_type == "flash"
        self.sparse = self.attention_type not in ("global", "flash")
        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            if self.use_flash_attention:
                from megatron.model.flash_attention import (
                    flash_attn_unpadded_qkvpacked_func_cuda,
                    flash_attn_unpadded_kvpacked_func_cuda,
                    flash_attn_unpadded_unpacked_func_triton,
                )

                self.flash_triton_fn = flash_attn_unpadded_unpacked_func_triton
                self.flash_qkv_fn = flash_attn_unpadded_qkvpacked_func_cuda
                self.flash_kv_fn = flash_attn_unpadded_kvpacked_func_cuda
            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            else:
                self.scale_mask_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=self.fp16,
                    input_in_bf16=self.bf16,
                    fusion_type=get_fusion_type(neox_args),
                    mask_func=self.attention_mask_func,
                    softmax_in_fp32=self.attention_softmax_in_fp32,
                    scale=None,
                )
            self.dropout_p = neox_args.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=neox_args.use_bias_in_attn_linear,
        )

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]
        # ===========================
        # Attention probs and dropout
        # ===========================

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if self.pos_emb != "alibi":

            # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
            key_layer = key_layer.transpose(0, 1).reshape(
                output_size[0] * output_size[3], 1, output_size[1], -1
            )
            value_layer = value_layer.transpose(0, 1).reshape(
                output_size[0] * output_size[3], 1, output_size[1], -1
            )

            batch_size = output_size[0]
            max_seqlen_q = output_size[2]
            max_seqlen_k = output_size[3]

            cu_seqlens_q = torch.arange(
                0,
                (batch_size + 1) * max_seqlen_q,
                step=max_seqlen_q,
                dtype=torch.int32,
                device=query_layer.device,
            )

            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * max_seqlen_k,
                step=max_seqlen_k,
                dtype=torch.int32,
                device=key_layer.device,
            )

            if not self.training:

                # [sq, b, np, hn] -> [b * sq, np, hn]
                query_layer = query_layer.transpose(0, 1).reshape(
                    output_size[0] * output_size[2], output_size[1], -1
                )

                # Combined k/v into [b * sk, 2, np, hn].
                kv = torch.concat([key_layer, value_layer], dim=1)

                output = self.flash_kv_fn(
                    query_layer,
                    kv,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=None,
                    causal=True,
                )

            else:

                # [sq, b, np, hn] -> [b * sq, 1, np, hn]
                query_layer = query_layer.transpose(0, 1).reshape(
                    output_size[0] * output_size[2], 1, output_size[1], -1
                )

                # Combined q/k/v into [b * s, 3, np, hn].
                qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

                output = self.flash_qkv_fn(
                    qkv,
                    cu_seqlens_q,
                    max_seqlen_q,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=None,
                    causal=True,
                )

            # [b * sq, np, hn] -> [b, sq, np, hn]
            matmul_result = output.view(
                output_size[0], output_size[2], output.shape[1], output.shape[2]
            )
            # [b, sq, np, hn] -> [b, np, sq, hn]
            matmul_result = matmul_result.transpose(1, 2)

        else:
            # [sq, b, np, hn] -> [b, sq, np, hn]
            sq = query_layer.size(0)
            b = query_layer.size(1)
            sk = key_layer.size(0)

            query_layer = query_layer.transpose(0, 1)
            key_layer = key_layer.transpose(0, 1)
            value_layer = value_layer.transpose(0, 1)

            bias = self.alibi_embed.bias(sq, sk, query_layer.device, query_layer.dtype)
            bias = bias.unsqueeze(0).tile((b, 1, 1, 1))

            matmul_result = self.flash_triton_fn(
                query_layer, key_layer, value_layer, bias=bias, causal=True
            )
            matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

    def forward(self, hidden_states, attention_mask):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )

        # ==================================
        # Cache key and value for inference
        # ==================================

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        if self.use_flash_attention:
            context_layer = self.flash_attention(query_layer, key_layer, value_layer)
        elif not self.sparse:
            context_layer = self.attention(
                query_layer, key_layer, value_layer, attention_mask
            )
        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, present]

        return output, bias


class ParallelTransformerLayer(nn.Module):
    """
        A single transformer layer.

        Transformer layer takes input with size [b, s, h] and returns an
        output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        use_cache=False,
    ):

        super().__init__()

        norm, eps = get_norm(neox_args)
        # Layernorm on the input data.
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.use_cache = use_cache

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.mlp_type = neox_args.mlp_type

        # Self attention.
        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            hidden_size=neox_args.hidden_size,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            use_cache=self.use_cache,
        )

        # Layernorm on the output of the attention layer.
        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)

        # MLP
        if neox_args.mlp_type == "regular":
            self.mlp = ParallelMLP(
                neox_args=neox_args,
                hidden_size=neox_args.hidden_size,
                ffn_dim=neox_args.ffn_dim,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        else:
            raise KeyError(neox_args.mlp_type)

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn


    def forward(self, x, attention_mask):
        bias_dropout_fn = self._get_bias_dropout()
        # x: [b, s, h]
        residual = x
        # x = x + attn(ln1(x))
        attention_output, attention_bias = self.attention(
            self.input_layernorm(x), attention_mask
        )
        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents
        with torch.enable_grad():
            if attention_bias is not None:
                # Use special bias_dropout_fn if we have a bias term from the above attention layer
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual),
                    residual=residual,
                    prob=self.hidden_dropout,
                )
            else:
                # Otherwise just apply dropout + residual
                attention_output = (
                    torch.nn.functional.dropout(
                        attention_output,
                        p=self.hidden_dropout,
                        training=self.training,
                    )
                    + residual
                )

        # output = x + mlp(ln2(x))
        mlp_output, mlp_bias = self.mlp(
            self.post_attention_layernorm(attention_output)
        )

        with torch.enable_grad():
            output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(attention_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                    )

        return output



class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        return super().forward(hidden_states, attention_mask), attention_mask



class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        logits, _ = super().forward(hidden_state)
        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.norm(args)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
