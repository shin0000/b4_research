import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import mlflow
import albumentations as A


class TargetToTensor(object):
    def __init__(self):
        pass
    def __call__(self, data):
        data = np.array(data).astype(np.float32)
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]
            if np.max(data) != 0:
                data = data / np.max(data)
            data = torch.from_numpy(data).type(torch.float)
        else:
            # print("Multi Label")
            data = torch.from_numpy(data).type(torch.float)
        return data
    
class TargetToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, data):
        data = np.array(data).astype(np.float32)
        if len(data.shape) == 2:
            if np.max(data) != 0:
                data = data / np.max(data)
        else:
            # print("Multi Label")
            data = torch.from_numpy(data).type(torch.float)
        return data

train_transform1 = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
    # A.HorizontalFlip(p=0.5),
    # A.RandomRotate90(p=1.0),
    A.Resize(300, 300),
    A.RandomCrop(width=256, height=256),
])

train_transform2 = A.Compose([
    A.Resize(256, 256)
])

train_transform3 = A.Compose([
    A.Resize(2048, 2048)
])

evaluate_transform = A.Compose([
    A.Resize(256, 256)
])

instruments_transform1 = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(scale_limit=0.3)
    # A.Flip(),
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90),
    # A.Resize(300, 300),
    # A.RandomCrop(width=256, height=256),
])
