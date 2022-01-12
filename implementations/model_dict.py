from torch import nn
from torch.nn import functional as F
from torch import optim

from .losses import *
from .metrics import *
from .models import *

Model = {
    "UNet": UNet,
    "DeepLabV3": DeepLabV3,
    "BayesianUNet": BayesianUNet,
    "AttentionUNet": AttentionUNet,
    "BayesianAttentionUNet": BayesianAttentionUNet,
}
Criterion = {
    "Focal": FocalLoss,
    "Dice": DiceLoss,
    "BCE": nn.BCEWithLogitsLoss,
    "DiceBCE": DiceBCELoss,
    "FocalDice": FocalDiceLoss,
    "FocalDiceBCE": FocalDiceBCELoss,
    "UncertaintyDice": UncertaintyDiceLoss,
    "UncertaintyDice2": UncertaintyDiceLoss2,
    "UncertaintyDice3": UncertaintyDiceLoss3
}
Optimizer = {
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
}
Metric = {
    "IoU": iou_pytorch_with_sigmoid,
    "Dice": dice_pytorch_with_sigmoid
}
