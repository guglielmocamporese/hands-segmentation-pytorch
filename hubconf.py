"""
Torch Hub script for accessing te hand segmentation model outside the repo.
"""

##################################################
# Imports
##################################################

dependencies = ['torch', 'pytorch_lightning']

from model import HandSegModel
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
        _download_file_from_google_drive('1w7dztGAsPHD_fl_Kv_a8qHL4eW92rlQg', 'checkpoint')
        model = model.load_from_checkpoint('./checkpoint/checkpoint.ckpt', *args, **kwargs)
    return model


import requests

def _download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

