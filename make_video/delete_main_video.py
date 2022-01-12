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
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import albumentations as A
import random
import shutil
import sys
import time
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from implementations import *

class cfg:
    device = "cuda:0"
    multi_gpu = True
    multi_device_ids = [0, 3]
    class model_info:
        model =  ""
        in_channels = 3
        out_channels = 1
        batch_size = 32
        save_model_path = "./model_weight/{}/model.pth"

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../../data_NuVAT/pancreas")

#[(START_FRAME, END_FRAME)]
PRED_DATA = {
    "Case001-1": [],
    "Case001-2": [(3165, 3665)],
    "Case002-1": [],
    "Case003-2": [],
    "Case004-1": [],
    "Case006-1": [],
    "Case007-1": [],
    "Case008-1": [],
    "Case008-2": [],
    "Case009-2": [],
    "Case010-2": [],
}

# data = ["Case008-2"]

pixel_organ = 223
p_color = 0.5

START_FRAME = 3165
END_FRAME = 3665
VIDEO_PATH = '../../accVideo/Case001/case001_2.MTS'
VIDEO_NAME = VIDEO_PATH.split("/")[-1].split(".")[0]

COLOR = [128, 128, 255]
HSV_CENTER = 75
HSV_WIDTH = 75
MODE = "HSV"

H_IMG_SIZE = 1080
W_IMG_SIZE = 1920
SAVE_VIDEO_PATH = "output_video/{}/{}.mp4"
SAVE_VIDEO_PATH2 = "output_video/{}/{}_{}_{}_{}.mp4"

def pred2output(X_data_orig, Y_pred, mode="RGB"):
    if mode == "RGB":
        Y_mask = np.where(Y_pred >= 0.5, 1, 0)
        Y_mask_color = np.array([Y_mask * c for c in COLOR]).transpose(1, 2, 0)
        output = X_data_orig.copy()
        output = np.where(Y_mask[..., np.newaxis] == 1, p_color*Y_mask_color+(1-p_color)*X_data_orig, X_data_orig).astype(np.uint8)
    elif mode == "HSV":
        Y_mask_pred = HSV_CENTER + (Y_pred - 0.5) * (2 * HSV_WIDTH)
        Y_mask_color = np.zeros((H_IMG_SIZE, W_IMG_SIZE, 3), dtype=np.uint8)
        Y_mask_color[:, :, 0] = 255
        Y_mask_hsv = cv2.cvtColor(Y_mask_color, cv2.COLOR_RGB2HSV_FULL)
        Y_mask_hsv[..., 0] = Y_mask_pred
        Y_mask = cv2.cvtColor(Y_mask_hsv, cv2.COLOR_HSV2RGB_FULL)
        output = p_color * Y_mask + (1-p_color) * X_data_orig
        output = output.astype(np.uint8)
    return output

def frame2tensor(frame):
    frame = A.Resize(256, 256)(image=frame)["image"]
    tensor = transforms.ToTensor()(frame)
    return tensor

def case2ifold(case):
    data = ["Case001-1", "Case001-2", "Case002-1", "Case003-2", "Case004-1", "Case006-1", "Case007-1", "Case008-1", "Case008-2", "Case009-2", "Case010-2"]
    folds = {
        0: {
            "train_data": [data[4], data[5], data[6], data[7], data[8], data[9], data[10]],
            "valid_data": [data[0], data[1], data[2], data[3]],
            "test_data": [data[0], data[1], data[2], data[3]],
        },
        1: {
            "train_data": [data[0], data[1], data[2], data[3], data[8], data[9], data[10]],
            "valid_data": [data[4], data[5], data[6], data[7]],
            "test_data": [data[4], data[5], data[6], data[7]],
        },
        2: {
            "train_data": [data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]],
            "valid_data": [data[8], data[9], data[10]],
            "test_data": [data[8], data[9], data[10]],
        },
    }
    i_fold = -1
    for i in range(len(folds)):
        if case in folds[i]["test_data"]:
            i_fold = i + 1
    if i_fold == -1:
        raise ValueError("Missing Data")
    return i_fold


# used_model_names = ["unet_aug1", "unet_aug3", "bunet_aug3!nd", "bunet_aug3!ed_10_5", "bunet_aug3!ed_10_3", "bunet_aug3!ed_10_1"]
# used_model_names = ["bunet_aug3!ed_10", "bunet_aug3!nd"]
used_model_names = ["bunet_aug3!ed_10_5"]
for used_model_name in used_model_names:
    print("+"*30)
    print(used_model_name)
    print("+"*30)
    img_outdir = './img'
    os.makedirs(img_outdir, exist_ok=True)

    if used_model_name.find("unet") == 0:
        cfg.model_info.model = "UNet"
        DROPOUT = False
    elif used_model_name.find("aunet") == 0:
        cfg.model_info.model = "AttentionUNet"
        DROPOUT = False
    elif used_model_name.find("bunet") == 0:
        cfg.model_info.model = "BayesianUNet"
        option = used_model_name.split("!")[1]
        if option == "nd":
            DROPOUT = False
            N_INFERENCE = 1
        else:
            DROPOUT = True
            N_INFERENCE = int(option.split("_")[1])
            DROPOUT_RATE = float(option.split("_")[2]) / 10 if len(option.split("_")) > 2 else 0.5
    elif used_model_name.find("baunet") == 0:
        cfg.model_info.model = "BayesianAttentionUNet"
        option = used_model_name.split("!")[1]
        if option == "nd":
            DROPOUT = False
            N_INFERENCE = 1
        else:
            DROPOUT = True
            N_INFERENCE = int(option.split("_")[1])
            DROPOUT_RATE = float(option.split("_")[2]) / 10 if len(option.split("_")) > 2 else 0.5
    else:
        raise ValueError("未実装")

    if cfg.model_info.model in ["UNet"]:
        model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
    elif cfg.model_info.model in ["BayesianUNet"]:
        model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels, DROPOUT_RATE)
    elif cfg.model_info.model in ["AttentionUNet"]:
        model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
    elif cfg.model_info.model in ["BayesianAttentionUNet"]:
        model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
    else:
        raise ValueError("未実装")

    if (cfg.device[:4] == "cuda") & (cfg.multi_gpu):
        model = nn.DataParallel(model, device_ids=cfg.multi_device_ids)

    if str(model.__class__) == "<class 'torch.nn.parallel.data_parallel.DataParallel'>":
        print("parallel")
    else:
        print("no parallel")
    model.load_state_dict(torch.load(cfg.model_info.save_model_path.format(used_model_name.split("!")[0])))
    

    model.eval()
    if DROPOUT:
        if str(model.__class__) == "<class 'torch.nn.parallel.data_parallel.DataParallel'>":
            model.module.enable_dropout()
        else:
            model.enable_dropout()

    print("Making Video")
    cap = cv2.VideoCapture(os.path.join(current_dir, VIDEO_PATH))
    if not cap.isOpened():
        raise ValueError("CAPUTER FAULT")
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    ####
    img_names = []
    inference_times = []
    with torch.no_grad():
        model = model.to(cfg.device)
        # for case, start_end_sets in PRED_DATA.items():
        #     for start_end_set in start_end_sets:
        for i in range(END_FRAME - START_FRAME):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                raise ValueError("Problem with frame")

            frame = resize9x16(frame)
            X_data_orig = frame.copy()
            if X_data_orig.shape[:2] != (H_IMG_SIZE, W_IMG_SIZE):
                print("Error")
                print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            X_data = frame2tensor(frame)
            X_data = X_data.to(cfg.device).unsqueeze(0)
            start_time = time.time()

            if cfg.model_info.model in ["BayesianUNet"]:
                Y_preds = []
                for j in range(N_INFERENCE):
                    Y_pred = model(X_data)
                    Y_preds.append(Y_pred)
                Y_pred = torch.stack(Y_preds).mean(axis=0)
                # Y_pred = torch.stack(Y_preds).median(axis=0).values
                # Y_pred = torch.stack(Y_preds).min(axis=0).values
            else:
                Y_pred = model(X_data)

            end_time = time.time()
            inference_times.append(end_time - start_time)
            Y_pred = Y_pred.cpu().squeeze().sigmoid().numpy()
            Y_pred = resize9x16(Y_pred)

            output = pred2output(X_data_orig, Y_pred, mode=MODE)

            img_name = '{}/{:09d}.png'.format(img_outdir, i)
            cv2.imwrite(img_name, output)

            img_names.append(img_name)
    ####
    print("Average Inference Time {}(s)".format(np.mean(inference_times)))

    # SAVE_VIDEO_NAME = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("20%y-%m-%d-%H-%M")
    img_names = os.listdir("img")
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    # video  = cv2.VideoWriter(SAVE_VIDEO_PATH.format(used_model_name, SAVE_VIDEO_NAME), fourcc, 20, (W_IMG_SIZE, H_IMG_SIZE))
    video  = cv2.VideoWriter(SAVE_VIDEO_PATH2.format(used_model_name.split("!")[0], used_model_name, VIDEO_NAME, START_FRAME, END_FRAME), fourcc, 20, (W_IMG_SIZE, H_IMG_SIZE))
    for img_name in img_names:
        img_name = "img/" + img_name
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)

    video.release()
    shutil.rmtree("img")