import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch import optim
import mlflow

# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        
    def forward(self, Y_pred, Y, Y_uncertainty=1, e=1): #not use Y_uncertainty 
        Y_pred = Y_pred.sigmoid()
        Y_pred = Y_pred.view(-1)
        Y = Y.view(-1)
        
        intersection = (Y_pred * Y).sum()
        union = Y_pred.sum() + Y.sum()
        dice = (2. * intersection + e) / (union + e)
        return 1 - dice

class UncertaintyDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(UncertaintyDiceLoss, self).__init__()
        
    def forward(self, Y_pred, Y, Y_uncertainty=1, e=1, p=0.7):
        Y_pred = Y_pred.sigmoid()
        Y_pred = Y_pred.view(-1)
        Y = Y.view(-1)
        if type(Y_uncertainty) != int:
            Y_uncertainty = Y_uncertainty.view(-1)
            Y_uncertainty = torch.where(Y_uncertainty < p, torch.tensor(p, dtype=Y_uncertainty.dtype).to(Y_uncertainty.device.type + ":" + str(Y_uncertainty.device.index)), Y_uncertainty) + (1-p)
            intersection = (Y_pred * Y * Y_uncertainty).sum()
        else:
            intersection = (Y_pred * Y).sum()

        union = Y_pred.sum() + Y.sum()
        dice = (2. * intersection + e) / (union + e)
        return 1 - dice

class UncertaintyDiceLoss2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(UncertaintyDiceLoss2, self).__init__()
        
    def forward(self, Y_pred, Y, Y_uncertainty=1, e=1, p=0.7):
        Y_pred = Y_pred.sigmoid()
        Y_pred = Y_pred.view(-1)
        Y = Y.view(-1)
        if type(Y_uncertainty) != int:
            Y_uncertainty = Y_uncertainty.view(-1)
            Y_uncertainty = torch.where(Y_uncertainty < p, torch.tensor(p, dtype=Y_uncertainty.dtype).to(Y_uncertainty.device.type + ":" + str(Y_uncertainty.device.index)), Y_uncertainty)
            intersection = (Y_pred * Y * Y_uncertainty).sum()
        else:
            intersection = (Y_pred * Y).sum()

        union = Y_pred.sum() + Y.sum()
        dice = (2. * intersection + e) / (union + e)
        return 1 - dice

class UncertaintyDiceLoss3(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(UncertaintyDiceLoss3, self).__init__()
        
    def forward(self, Y_pred, Y, Y_uncertainty=1, e=1, p=0.7):
        Y_pred = Y_pred.sigmoid()
        Y_pred = Y_pred.view(-1)
        Y = Y.view(-1)
        if type(Y_uncertainty) != int:
            Y_uncertainty = Y_uncertainty.view(-1)
            Y_uncertainty = torch.where(Y_uncertainty < p, torch.tensor(p, dtype=Y_uncertainty.dtype).to(Y_uncertainty.device.type + ":" + str(Y_uncertainty.device.index)), Y_uncertainty)
            intersection = (Y_pred * Y * Y_uncertainty).sum()
        else:
            intersection = (Y_pred * Y).sum()

        union = Y_pred.sum() + Y.sum()
        dice = (2. * intersection + e) / (union + e)
        return 1 - dice

class UncertaintyESDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(UncertaintyDiceLoss, self).__init__()
        
    def forward(self, Y_pred, Y, Y_uncertainty=1, Y_uncertainty_std=1, e=1, p=0.7):
        Y_pred = Y_pred.sigmoid()
        Y_pred = Y_pred.view(-1)
        Y = Y.view(-1)
        if type(Y_uncertainty) != int:
            Y_uncertainty = Y_uncertainty.view(-1)
            Y_uncertainty_std = Y_uncertainty_std.view(-1)
            Y_uncertainty = torch.where(Y_uncertainty < p, torch.tensor(p, dtype=Y_uncertainty.dtype).to(Y_uncertainty.device.type + ":" + str(Y_uncertainty.device.index)), Y_uncertainty) + (1-p)
            intersection = (Y_pred * Y * Y_uncertainty * Y_uncertainty_std).sum()
        else:
            intersection = (Y_pred * Y).sum()

        union = Y_pred.sum() + Y.sum()
        dice = (2. * intersection + e) / (union + e)
        return 1 - dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        
    def forward(self, Y_pred, Y, e=1):
        Y_pred = Y_pred.sigmoid()
        Y_pred = Y_pred.view(-1)
        
        Y = Y.view(-1)
        intersection = (Y_pred * Y).sum()
        union = Y_pred.sum() + Y.sum()
        dice = (2. * intersection + e) / (union + e)
        dice_loss = 1 - dice
        bce_loss = F.binary_cross_entropy(Y_pred, Y, reduction='mean')
        Dice_BCE = bce_loss + dice_loss
        return Dice_BCE
    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_averate=True):
        super(FocalLoss, self).__init__()
        
    def forward(self, Y_pred, Y, alpha=0.8, gamma=2, e=1):
        Y_pred = Y_pred.sigmoid()
        Y_pred = Y_pred.view(-1)
        Y = Y.view(-1)
        
        BCE = F.binary_cross_entropy(Y_pred, Y, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        return focal_loss

class FocalDiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalDiceBCELoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, Y_pred, Y):
        return self.focal_loss(Y_pred, Y) + self.dice_loss(Y_pred, Y) + self.bce_loss(Y_pred, Y)

class FocalDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, Y_pred, Y):
        return self.focal_loss(Y_pred, Y) + self.dice_loss(Y_pred, Y)
