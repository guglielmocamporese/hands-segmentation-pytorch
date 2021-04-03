# Hands Segmentation in PyTorch - A Plug and Play Model

## Example of predictions on the test set
![alt text](test_preds.png "Title")
The model checkpoint reaches 0.904 of mIoU on the test set.

# Datasets
In this project I considered the following datasets for training the model:
- EgoHands
- EgoYouTubeHands (EYTH) 
- GTEA
- HandOverFace (HOF)

I set up a script `scripts/download_datasets.sh` that downloads and prepares all the previous datasets into the `DATA_BASE_PATH` folder, specified in the script.


# Model
I used the [PyTorch implementation](https://pytorch.org/vision/stable/models.html#semantic-segmentation) of  [DeepLabV3](https://arxiv.org/abs/1706.05587) with ResNet50 backbone.

# Train
An example of script used for training the model is reported in `script/train.sh`, and reported here:

```bash
#!/bin/bash

python main.py \
    --mode train \
    --epochs 50 \
    --batch_size 16 \
    --gpus 1 \
    --datasets 'eyth eh hof gtea' \
    --height 256 \
    --width 256 \
    --data_base_path './data' \
    --model_pretrained
```

# Finetuning
An example of script used for finetuning the model is reported in `script/finetune.sh`, and reported here:

```bash
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
```

# Test
An example of script used for testing the model is reported in `script/test.sh`, and reported here:

```bash
#!/bin/bash

python main.py \
    --mode test \
    --data_base_path "./data" \
    --model_pretrained \
    --model_checkpoint "./checkpoint/checkpoint.ckpt"
```

# Predict From a Custom Dataset
With this code you can do inference and compute the predictions starting from a set of custom images, you just have to specify the folder that contains the images in the variable `data_base_path` in the `script/predict.sh` script.

You can find an example of a script used for predicting at `script/predict.sh`, and reported here: 

```bash
#!/bin/bash

python main.py \
    --mode predict \
    --data_base_path './test_images' \
    --model_checkpoint "./checkpoint/checkpoint.ckpt" \
    --model_pretrained
```
