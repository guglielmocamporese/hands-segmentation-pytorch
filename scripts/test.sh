#!/bin/bash

python main.py \
    --mode test \
    --data_base_path "./data" \
    --model_pretrained \
    --model_checkpoint "./checkpoint/checkpoint.ckpt"
