#!/bin/bash

python main.py \
    --mode predict \
    --data_base_path './test_images' \
    --model_checkpoint "./checkpoint/checkpoint.ckpt" \
    --model_pretrained
