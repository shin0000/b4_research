import numpy as np
import pandas as pd
import scipy as sp
# import seaborn as sns
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
    device = "cuda:1"
    multi_gpu = True
    multi_device_ids = [1, 3]
    class model_info:
        model =  ""
        in_channels = 3
        out_channels = 1
        batch_size = 32
        save_model_path = "./model_weight/{}/{}/model.pth"
        dropout_rate = 0.3

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../../data_NuVAT/pancreas")

PRED_DATA = {
    "Case001-1": ["176343"],
    "Case001-2": ["000428"],
    "Case002-1": ["207239"],
    "Case003-2": ["008433"],
    "Case004-1": ["147509"],
    "Case006-1": ["208927"],
    "Case007-1": ["122401"],
    "Case008-1": ["217513"],
    "Case008-2": ["003041"],
    "Case009-2": ["108365"],
    "Case010-2": ["152612"],
}

# PRED_DATA = {
#     "Case001-1": ["176343"],
# }

pixel_organ = 223
p_color = 0.5

FRAME_FORMAT = "movieFrame_{}.png"
LABEL_FORMAT = "label_{}.png"

START_FRAME = 3165
END_FRAME = 3665

VIDEO_PATH = '../../accVideo/Case001/case001_2.MTS'
CASE_NAME = VIDEO_PATH.split("/")[-2]
VIDEO_NAME = VIDEO_PATH.split("/")[-1].split(".")[0]

COLOR = [128, 128, 255]
# COLOR = [255, 128, 128] #BGR
HSV_CENTER = 75
HSV_WIDTH = 75
MODE = "RGB"

H_IMG_SIZE = 1080
W_IMG_SIZE = 1920
SAVE_VIDEO_PATH = "output_video/{}/{}.mp4"
SAVE_VIDEO_PATH2 = "output_video/{}/{}_{}_{}_{}.mp4"

TIME_NAME = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("20%y-%m-%d-%H-%M")

def pred2output(X_data_orig, Y_pred, mode="RGB"):
    if mode == "RGB":
        Y_mask = np.where(Y_pred >= 0.2, 1, 0)
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

img_dir = './output_img/{}'.format(TIME_NAME)

try:
    os.makedirs(img_dir)
except:
    pass

# used_model_names = ["unet_aug1", "unet_aug3", "bunet_aug3!nd", "bunet_aug3!ed_10_5", "bunet_aug3!ed_10_3", "bunet_aug3!ed_10_1"]
# used_model_names = ["bunet_aug3!ed_10", "bunet_aug3!nd"]
used_model_names = ["label", "unet_aug1", "unet_aug5", "aunet_aug1", "aunet_aug5", "bunet_aug1!ed_10", "bunet_aug5!ed_10", "baunet_aug1!ed_10", "baunet_aug5!ed_10"]
img_paths = []
for used_model_name in used_model_names:
    print("+"*30)
    print(used_model_name)
    print("+"*30)
    if used_model_name == "label":
        img_outdir = os.path.join(img_dir, used_model_name)
        try:
            os.mkdir(img_outdir)
        except:
            pass
        for case, frame_numbers in PRED_DATA.items():
            for frame_number in frame_numbers:
                frame_name = FRAME_FORMAT.format(frame_number)
                frame_path = os.path.join(data_dir, case, "movie", frame_name)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                label_name = LABEL_FORMAT.format(frame_number)
                label_path = os.path.join(data_dir, case, "label", label_name)
                label = cv2.imread(label_path)[..., 0]
                label = label / np.max(label)

                output = pred2output(frame, label, mode=MODE)
                #####
                case_dir = "{}/{}".format(img_outdir, case)
                try:
                    os.mkdir(case_dir)
                except:
                    pass
                img_path = "{}/{}".format(case_dir, frame_name)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, output)

                img_paths.append(img_path)
        print("Done!")
        continue
    #####

    img_outdir = os.path.join(img_dir, used_model_name.split("!")[0])
    try:
        os.mkdir(img_outdir)
    except:
        pass

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

    ###
    print("Making Images")
    inference_times = []
    with torch.no_grad():
        for case, frame_numbers in PRED_DATA.items():
            i_fold = case2ifold(case)
            model.load_state_dict(torch.load(cfg.model_info.save_model_path.format(used_model_name.split("!")[0], "{}_fold".format(i_fold))))
            model.eval()
            if DROPOUT:
                if str(model.__class__) == "<class 'torch.nn.parallel.data_parallel.DataParallel'>":
                    model.module.enable_dropout()
                else:
                    model.enable_dropout()
            model = model.to(cfg.device)

            for frame_number in frame_numbers:
                frame_name = FRAME_FORMAT.format(frame_number)
                frame_path = os.path.join(data_dir, case, "movie", frame_name)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #####
                frame = resize9x16(frame)
                X_data_orig = frame.copy()
                if X_data_orig.shape[:2] != (H_IMG_SIZE, W_IMG_SIZE):
                    print("Error")
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
                #####
                case_dir = "{}/{}".format(img_outdir, case)
                try:
                    os.mkdir(case_dir)
                except:
                    pass
                img_path = "{}/{}".format(case_dir, frame_name)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, output)

                img_paths.append(img_path)
    ####
    print("Average Inference Time {}(s)".format(np.mean(inference_times)))

for case, frame_numbers in PRED_DATA.items():
    for frame_number in frame_numbers:
        fig = plt.figure(figsize=(len(used_model_names) * 16, 9))
        for i, used_model_name in enumerate(used_model_names):
            # if used_model_name == "label":
            #     frame_name = LABEL_FORMAT.format(frame_number)
            #     frame_path = os.path.join(data_dir, case, "label", frame_name)
            #     frame = cv2.imread(frame_path)
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # else:
            frame_name = FRAME_FORMAT.format(frame_number)
            frame_path = os.path.join(img_dir, used_model_name.split("!")[0], case, frame_name)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ax = fig.add_subplot(1, len(used_model_names), i+1)
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
        save_path = os.path.join(img_dir, case + "_" + frame_name)
        plt.savefig(save_path)

