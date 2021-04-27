"""
Torch Hub script for accessing te hand segmentation model outside the repo.
"""

##################################################
# Imports
##################################################

from model import HandSegModel
import os

dependencies = ['torch', 'pytorch_lightning', 'pytorch-lightning']


def hand_segmentor(pretrained=True, *args, **kwargs):
    """
    Hand segmentor based on a DeepLabV3 model with a ResNet50 encoder.
    DeeplabV3: https://arxiv.org/abs/1706.05587
    ResNet50: https://arxiv.org/abs/1512.03385
    """
    model = HandSegModel(*args, **kwargs)
    if pretrained:
        os.system('chmod +x ./scripts/download_model_checkpoint.sh')
        os.system('./scripts/download_model_checkpoint.sh')
        model = model.load_from_checkpoint('./checkpoint/checkpoint.ckpt', *args, **kwargs)
    return model

