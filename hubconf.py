"""
Torch Hub script for accessing te hand segmentation model outside the repo.
"""

##################################################
# Imports
##################################################

dependencies = ['torch', 'pytorch_lightning']

from model import HandSegModel
import gdown
import os



def hand_segmentor(pretrained=True, *args, **kwargs):
    """
    Hand segmentor based on a DeepLabV3 model with a ResNet50 encoder.
    DeeplabV3: https://arxiv.org/abs/1706.05587
    ResNet50: https://arxiv.org/abs/1512.03385
    """
    model = HandSegModel(*args, **kwargs)
    if pretrained:
        #os.system('chmod +x ./scripts/download_model_checkpoint.sh')
        #os.system('./scripts/download_model_checkpoint.sh')
        _download_file_from_google_drive('1w7dztGAsPHD_fl_Kv_a8qHL4eW92rlQg', './checkpoint/checkpoint.ckpt')
        model = model.load_from_checkpoint('./checkpoint/checkpoint.ckpt', *args, **kwargs)
    return model


def _download_file_from_google_drive(id, destination):

    url = f'https://drive.google.com/uc?id={id}'
    path = os.path.dirname(destination)
    if not os.path.exists(path):
        os.makedirs(path)
    gdown.download(url, destination, quiet=False)
