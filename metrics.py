##################################################
# Imports
##################################################

import numpy as np
import torch
import torch.nn.functional as F


##################################################
# Mean IoU
##################################################

def meanIoU(logits, labels):
    """
    Computes the mean intersection over union (mIoU).
    
    Args:
        logits: tensor of shape [bs, c, h, w].
        labels: tensor of shape [bs, h, w].
    
    Output:
        miou: scalar.
    """
    num_classes = logits.shape[1]
    preds = F.softmax(logits, 1)
    preds_oh = F.one_hot(preds.argmax(1), num_classes).permute(0, 3, 1, 2).to(torch.float32) # [bs, c, h, w] 
    labels_oh = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).to(torch.float32) # [bs, c, h, w]
    tps = (preds_oh * labels_oh).sum(-1).sum(-1) # true positives [bs, c]
    fps = (preds_oh * (1 - labels_oh)).sum(-1).sum(-1) # false positives [bs, c]
    fns = ((1 - preds_oh) * labels_oh).sum(-1).sum(-1) # false negatives [bs, c]
    iou = tps / (tps + fps + fns + 1e-8) # [bs, c]
    return iou.mean(-1).mean(0)
