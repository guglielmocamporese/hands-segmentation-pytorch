##################################################
# Imports
##################################################

import argparse
import json
from torchvision import transforms
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm

# Custom
from model import HandSegModel
from dataloader import get_dataloader, show_samples, Denorm

def get_args():
    """
    read the input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', 
                        help='Mode of the program. Can be "train", "test" or "predict".')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs used for the training.')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size.')
    parser.add_argument('--gpus', type=int, default=1, help='The number of gpus used.')
    parser.add_argument('--datasets', type=str, default='eyth eh hof gtea', help='List of datasets to use.')
    parser.add_argument('--height', type=int, default=256, help='The height of the input image.')
    parser.add_argument('--width', type=int, default=256, help='THe width of the input image.')
    parser.add_argument('--data_base_path', type=str, required=True, help='The path of the input dataset.')
    parser.add_argument('--model_pretrained', default=False, action='store_true', 
                        help='Load the PyTorch pretrained model.')
    parser.add_argument('--model_checkpoint', type=str, default='', help='The model checkpoint to load.')
    parser.add_argument('--lr', type=float, default=3e-4, help='The learning rate.')
    parser.add_argument('--in_channels', type=int, default=3, choices=[1, 3, 4],
                        help='The number of input channels (3 for RGB, 1 for Grayscale, 4 for RGBD).')
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    return args

def get_model(args):
    """
    build the model.
    """
    model_args = {
        'pretrained': args.model_pretrained,
        'lr': args.lr,
        'in_channels': args.in_channels,
    }
    model = HandSegModel(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, **model_args)
        print(f'Loaded checkpoint from {args.model_checkpoint}.')
    return model

def get_image_transform(args):
    """
    build the image transforms.
    """
    image_transform = None
    pad_rgb2rgbd = lambda x: torch.cat([x, torch.zeros(3, x.shape[1], x.shape[2])], 0)
    pad_gray2rgbd = lambda x: torch.cat([x.repeat(3, 1, 1), torch.zeros(3, x.shape[1], x.shape[2])], 0)
    def to_rgbd(x):
        C = x.shape[0]
        if C == 4: return x
        elif C == 3: return pad_rgb2rgbd(x)
        elif C == 1: return pad_gray2rgbd(x)
    if args.in_channels == 1:
        image_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
            lambda x: x.mean(0, keepdims=True), # convert RGB into grayscale
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    elif args.in_channels == 3:
        image_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.in_channels == 4:
        image_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            lambda x: x if x.shape[0] == 4 else to_rgbd(x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
        ])
    return image_transform

def get_dataloaders(args):
    """
    build the dataloaders.
    """
    image_transform = get_image_transform(args)
    mask_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        lambda m: torch.where(m > 0, torch.ones_like(m), torch.zeros_like(m)),
        lambda m: F.one_hot(m[0].to(torch.int64), 2).permute(2, 0, 1).to(torch.float32),
    ])
    dl_args = {
        'data_base_path': args.data_base_path,
        'datasets': args.datasets.split(' '),
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'batch_size': args.batch_size,
    }
    dl_train = get_dataloader(**dl_args, partition='train', shuffle=True)
    dl_validation = get_dataloader(**dl_args, partition='validation', shuffle=False)
    dl_test = get_dataloader(**dl_args, partition='test', shuffle=False)
    dls = {
        'train': dl_train,
        'validation': dl_validation,
        'test': dl_test,
    }
    return dls

def get_predict_dataset(args):
    """
    """
    image_paths = sorted(os.listdir(args.data_base_path))
    image_paths = [os.path.join(args.data_base_path, f) for f in image_paths if not f.endswith("-pred.png")]
    print(f'Found {len(image_paths)} in {args.data_base_path}.')
    transform = get_image_transform(args)
    class ImageDataset(Dataset):
        def __init__(self, image_paths, transform=None):
            super(ImageDataset, self).__init__()
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            W, H = image.size
            sample = {
                "image": image,
                "height": H,
                "width": W,
            }
            if self.transform is not None:
                sample["image"] = self.transform(sample["image"])
            return sample, image_path
    return ImageDataset(image_paths, transform=transform)

def main(args):
    """
    main function.
    """

    # Model
    model = get_model(args)
    print("model built")

    # Mode
    if args.mode == 'train':
        dls = get_dataloaders(args) # Dataloader
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus)
        trainer.fit(model, dls['train'], dls['validation'])
    elif args.mode == 'validation':
        dls = get_dataloaders(args) # Dataloader
        trainer = pl.Trainer(gpus=args.gpus)
        trainer.test(model, dls['validation'])
    elif args.mode == 'test':
        dls = get_dataloaders(args) # Dataloader
        trainer = pl.Trainer(gpus=args.gpus)
        trainer.test(model, dls['test'])
    elif args.mode == 'predict':
        ds = get_predict_dataset(args) # Dataset

        # Save prediction
        _ = model.eval()
        device = next(model.parameters()).device
        for sample, x_path in tqdm(ds, desc='Save predictions'):
            out_path = x_path[:-4]
            x = sample["image"]
            H, W = sample["height"], sample["width"]
            x = x.unsqueeze(0).to(device)
            logits = model(x).detach().cpu()
            preds = F.softmax(logits, 1).squeeze(0)[1] * 255 # [h, w]
            preds = Image.fromarray(preds.numpy().astype(np.uint8), 'L')
            preds = preds.resize((W, H), resample=Image.BICUBIC)
            preds.save(f'{out_path}-pred.png')
    else:
        raise Exception(f'Error. Mode "{args.mode}" is not supported.')


##################################################
# Main
##################################################

if __name__ == '__main__':
    args = get_args()
    main(args)
