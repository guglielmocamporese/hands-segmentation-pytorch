
# Hands Segmentation in PyTorch - A Plug and Play Model

If you need hands segmentation for your project, you are in the correct place!

## What you can do with this code
This code provides:
- A collection of **4** different **datasets** for hands segmentation (see the **Datasets** section for more details), that can be use for train a hands segmentation model,

-  **Train** and **evaluate** a hand segmentation model,

- **Finetune** a our provided  model for hand segmentation on a custom dataset,

-  **Get hands segmentation maps** on unseen (your) custom data, using our pretrained (or your) model.

  

##### Example of Predictions on the Test Set

![alt text](test_preds.png "Title")

  

##### Performance

The model checkpoint reaches `0.904` of mIoU on the test set.


## Install
Once you have cloned the repo, all the commands below should be runned inside the main project folder  `hands` folder:

```python
# Clone the repo
$ git clone https://github.com/guglielmocamporese/hands-segmentation-pytorch.git hands

# Go inside the project folder
$ cd hands
```
To run the code you need to have conda installed (version >= 4.9.2).

Furthermore, all the requirements for running the code are specified in the  `environment.yaml`  file and can be installed with:

```
# Install the conda env
$ conda env create --file environment.yaml

# Activate the conda env
$ conda activate hands
```



  

# Datasets

In this project I considered the following datasets for training the model:

- #### **EgoHands** [[link]](http://vision.soic.indiana.edu/projects/egohands/)

- **4800** labeled frames (**100** labeled frames from **48** different videos),

- each frame is **720**x**1280**,

- **1.3** GB of zip file,

- #### **EgoYouTubeHands (EYTH)** [[link]](https://github.com/aurooj/Hand-Segmentation-in-the-Wild)

- **774** labeled frames,

- each frame is **216**x**384**,

- **17** MB of tar.gz file,

- #### **GTEA (with GTEA GAZE PLUS)** [[link]](http://cbs.ic.gatech.edu/fpv/)

- **1067** labeled frames,

- each frame of GTEA is **405**x**720**, each frame of GTEA GAZE PLUS is **720**x**960**,

- **250** MB of zip file,

- #### **HandOverFace (HOF)** [[link]](https://github.com/aurooj/Hand-Segmentation-in-the-Wild)

- **180** labeled frames,

- each frame is **384**x**216**,

- **41** MB of tar.gz file,

  

I set up a script `scripts/download_datasets.sh` that downloads and prepares all the previous datasets into the `DATA_BASE_PATH` folder, specified in the script itself.

  

# Model

I used the [PyTorch implementation](https://pytorch.org/vision/stable/models.html#semantic-segmentation) of [DeepLabV3](https://arxiv.org/abs/1706.05587) with ResNet50 backbone. In particular I trained the model for hands segmentation starting from the pretrained DeepLabV3 on COCO train2017.

We provide the code for downloading our model checkpoint:
```python
# Download our pre-trained model
$ ./scripts/download_model_checkpoint.sh
```
This will download the checkpoint `checkpoint.ckpt` inside the `./checkpoint` folder.

  

# Train

An example of script used for training the model is reported in `scripts/train.sh` and reported here:

  

```bash
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

An example of script used for finetuning the model is reported in `scripts/finetune.sh` and reported here:

  

```bash
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

An example of script used for testing the model is reported in `scripts/test.sh` and reported here:

  

```bash
python main.py \
	--mode test \
	--data_base_path "./data" \
	--model_pretrained \
	--model_checkpoint "./checkpoint/checkpoint.ckpt"
```

  

# Predict From a Custom Dataset

With this code you can do inference and compute the predictions starting from a set of custom images, you just have to specify the folder that contains the images in the variable `data_base_path` in the `scripts/predict.sh` script.

  

Each prediction computed from the image `path/to/image.jpg` will be saved at `path/to/image.jpg.png`.

  

You can find an example of a script used for predicting at `scripts/predict.sh`. I also reported it here:

  

```bash
python main.py \
	--mode predict \
	--data_base_path './test_images' \
	--model_checkpoint "./checkpoint/checkpoint.ckpt" \
	--model_pretrained
```
