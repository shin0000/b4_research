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

# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def iou_pytorch_with_sigmoid(Y_pred, Y, threshold=0.5, e=1e-6):
    Y = Y.long().squeeze(1)
    Y_pred = Y_pred.sigmoid().squeeze(1)
    Y_pred = torch.where(Y_pred > threshold, 1, 0).long()
    
    intersection = (Y_pred & Y).float().sum((1, 2))
    union = (Y_pred | Y).float().sum((1, 2))
    
    iou = (intersection + e) / (union + e)
    iou = torch.mean(iou)
    return iou

def dice_pytorch_with_sigmoid(Y_pred, Y, threshold=0.5, e=1e-6):
    Y = Y.long().squeeze(1)
    Y_pred = Y_pred.sigmoid().squeeze(1)
    Y_pred = torch.where(Y_pred > threshold, 1, 0).long()
    
    intersection = (Y_pred & Y).float().sum((1, 2))
    pred_sum = Y_pred.sum((1, 2))
    target_sum = Y.sum((1, 2))
    
    dice = (2. * intersection + e) / (pred_sum + target_sum + e) 
    dice = torch.mean(dice)
    return dice

def f_pytorch_with_sigmoid(Y_pred, Y, threshold=0.5, e=1e-6):
    Y = Y.long().squeeze(1)
    Y_pred = Y_pred.sigmoid().squeeze(1)
    Y_pred = torch.where(Y_pred > threshold, 1, 0).long()
    
    intersection = (Y_pred & Y).float().sum((1, 2))
    pred_sum = Y_pred.sum((1, 2))
    target_sum = Y.sum((1, 2))
    
    f = (2. * intersection + e) / (pred_sum + target_sum + e) 
    f = torch.mean(f)
    return f.item()

def precision_pytorch_with_sigmoid(Y_pred, Y, threshold=0.5, e=1e-6):
    Y = Y.long().squeeze(1)
    Y_pred = Y_pred.sigmoid().squeeze(1)
    Y_pred = torch.where(Y_pred > threshold, 1, 0).long()
    
    intersection = (Y_pred & Y).float().sum((1, 2))
    pred_sum = Y_pred.sum((1, 2))
    
    precision = (intersection + e) / (pred_sum + e) 
    precision = torch.mean(precision)
    return precision.item()

def recall_pytorch_with_sigmoid(Y_pred, Y, threshold=0.5, e=1e-6):
    Y = Y.long().squeeze(1)
    Y_pred = Y_pred.sigmoid().squeeze(1)
    Y_pred = torch.where(Y_pred > threshold, 1, 0).long()
    
    intersection = (Y_pred & Y).float().sum((1, 2))
    target_sum = Y.sum((1, 2))
    
    recall = (intersection + e) / (target_sum + e) 
    recall = torch.mean(recall)
    return recall.item()

def sensitivity_pytorch_with_sigmoid(Y_pred, Y, e=1e-6):
    # Y_pred (0, 1), Y (0, 1)
    intersection = (Y_pred & Y).float().sum((1, 2))
    target_sum = Y.sum((1, 2))
    
    sensitivity = (intersection + e) / (target_sum + e) 
    sensitivity = torch.mean(sensitivity)
    return sensitivity.item()

def specificity_pytorch_with_sigmoid(Y_pred, Y, e=1e-6):
    # Y_pred (0, 1), Y (0, 1)
    intersection2 = (abs(Y_pred - 1) & abs(Y - 1)).float().sum((1, 2))
    target_sum2 = abs(Y - 1).sum((1, 2))
    
    specificity = (intersection2 + e) / (target_sum2 + e) 
    specificity = torch.mean(specificity)
    return specificity.item()

def roc_pytorch_with_sigmoid(Y_pred, Y, n_threshold=100, e=1e-6):
    Y = Y.long().squeeze(1)
    Y_pred = Y_pred.sigmoid().squeeze(1)

    roc_z = []
    for i in range(n_threshold):
        threshold = i / n_threshold
        Y_pred_ = torch.where(Y_pred > threshold, 1, 0).long()
        specificity = specificity_pytorch_with_sigmoid(Y_pred_, Y)
        sensitivity = sensitivity_pytorch_with_sigmoid(Y_pred_, Y)
    
        roc_z.append((1-specificity, sensitivity))

    roc_z = sorted(roc_z, key=lambda z: z[0])
    roc_x = [0.0] + [x for (x, y) in roc_z] + [1.0] #ここらへんで原因あるとしたらここ
    roc_y = [0.0] + [y for (x, y) in roc_z] + [1.0]

    return (roc_x, roc_y)

        

