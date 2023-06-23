#!/bin/bash

# download and tokenize OpenWebText2 dataset
# python prepare_data.py -d /home/youpengzhao/code/dataset/nlp openwebtext2


# train GPT-2 on OpenWebText2
python deepy.py train.py ./configs/125M.yml ./configs/local_setup.yml