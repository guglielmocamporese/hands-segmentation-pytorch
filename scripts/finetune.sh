#!/bin/bash

python main.py \
    --mode train \
    --epochs 10 \
    --batch_size 16 \
    --gpus 1 \
    --datasets 'eyth eh hof gtea' \
    --height 256 \
    --width 256 \
    --data_base_path './data' \
    --model_checkpoint "./checkpoint/checkpoint.ckpt"
    --model_pretrained
