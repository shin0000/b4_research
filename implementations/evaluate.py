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

from .metrics import *
from .losses import *

def evaluate_model(model, criterion, metricfn, test_dataloader, device, mlfr, i_fold=1, n_inference=1):
    model = model.to(device)
    model.eval()
    if n_inference > 1:
        if str(model.__class__) == "<class 'torch.nn.parallel.data_parallel.DataParallel'>":
            model.module.enable_dropout()
        else:
            model.enable_dropout()
    with torch.no_grad():
        Y_tests = []
        Y_test_preds = []
        for X_test, Y_test in test_dataloader:
            X_test, Y_test = X_test.to(device), Y_test.to(device)
            Y_test_preds_n = []
            for j in range(n_inference):
                Y_test_pred = model(X_test)
                Y_test_preds_n.append(Y_test_pred)
            Y_test_pred = torch.stack(Y_test_preds_n).mean(axis=0)
            Y_tests.append(Y_test)
            Y_test_preds.append(Y_test_pred)
    Y_test = torch.cat(Y_tests)
    Y_test_pred = torch.cat(Y_test_preds)
    loss = criterion(Y_test_pred, Y_test)
    metric = metricfn(Y_test_pred, Y_test)
    f = f_pytorch_with_sigmoid(Y_test_pred, Y_test)
    precision = precision_pytorch_with_sigmoid(Y_test_pred, Y_test)
    recall =recall_pytorch_with_sigmoid(Y_test_pred, Y_test)
    roc = roc_pytorch_with_sigmoid(Y_test_pred, Y_test)
    auc = auc_using_roc_curve(roc)
    save_roc_curve(roc, i_fold=i_fold)

    print("test: loss {}, metric {}".format(loss, metric))
    mlfr.log_metric("Test Loss {} fold".format(i_fold), loss.item())
    mlfr.log_metric("Test Metrics {} fold".format(i_fold), metric.item())
    mlfr.log_metric("Test F {} fold".format(i_fold), f)
    mlfr.log_metric("Test Precision {} fold".format(i_fold), precision)
    mlfr.log_metric("Test Recall {} fold".format(i_fold), recall)
    mlfr.log_metric("Test AUC {} fold".format(i_fold), auc)
    return loss.item(), metric.item(), f, precision, recall, auc

def compare_pred_label(model, test_dataset, test_dataset_orig, device, mlfr, i=0, key=None, i_fold=1, n_inference=1):
    model = model.to(device)
    model.eval()
    if n_inference > 1:
        if str(model.__class__) == "<class 'torch.nn.parallel.data_parallel.DataParallel'>":
            model.module.enable_dropout()
        else:
            model.enable_dropout()
    with torch.no_grad():
        X_test, Y_test = test_dataset[i]
        X_test_orig, Y_test_orig, _ = test_dataset_orig[i]
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        Y_test_preds = []
        for j in range(n_inference):
            Y_test_pred = model(X_test.unsqueeze(0)).squeeze().sigmoid()
            Y_test_preds.append(Y_test_pred)
        Y_test_pred = torch.stack(Y_test_preds).mean(axis=0).cpu().numpy()
    if len(Y_test_pred.shape) > 2:
        plot_image_with_multiclasspred(X_test_orig, Y_test_orig, Y_test_pred, mlfr, i, key, threshold = 0.5, i_fold=i_fold)
    else:
        plot_image_with_pred(X_test_orig, Y_test_orig, Y_test_pred, mlfr, i, threshold = 0.5, i_fold=i_fold)

def plot_image_with_pred(X_data, Y_data, Y_pred, mlfr, i=0, threshold = 0.5, i_fold=1):
    plt.figure(figsize=(48, 18))

    plt.subplot(2, 3, 1)
    plt.imshow(resize9x16(X_data))
    # plt.title("Original Image", fontsize=15)
    plt.xticks([])
    plt.yticks([])
    # plt.ylabel("Ground Truth", fontsize=15)

    plt.subplot(2, 3, 2)
    plt.imshow(resize9x16(Y_data))
    # plt.title("Probability Map", fontsize=15)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 3)
    Y_data = np.array(Y_data)
    if len(Y_data.shape) == 2:
        mask = np.zeros((*Y_data.shape, 3))
        mask[:, :, 0], mask[:, :, 1], mask[:, :, 2] = Y_data, Y_data, Y_data
        plt.imshow(resize9x16(np.where(mask != 0, X_data, 0)))
    elif len(Y_data.shape) == 3:
        plt.imshow(resize9x16(np.where(Y_data != 0, X_data, 0)))
    else:
        plt.imshow(resize9x16(np.where(Y_data != 0, X_data, 0)))
    # plt.title("Extraction Area", fontsize=15)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.imshow(resize9x16(X_data))
    plt.xticks([])
    plt.yticks([])
    # plt.ylabel("Prediction", fontsize=15)

    plt.subplot(2, 3, 5)
    plt.imshow(resize9x16(Y_pred))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 6)
    Y_pred = np.array(Y_pred)
    Y_pred = np.where(Y_pred > threshold, 1, 0)
    if len(Y_pred.shape) == 2:
        mask = np.zeros((*Y_pred.shape, 3))
        mask[:, :, 0], mask[:, :, 1], mask[:, :, 2] = Y_pred, Y_pred, Y_pred
        plt.imshow(resize9x16(np.where(mask != 0, X_data, 0)))
    elif len(Y_data.shape) == 3:
        plt.imshow(resize9x16(np.where(Y_data != 0, X_data, 0)))
    else:
        plt.imshow(resize9x16(np.where(Y_data != 0, X_data, 0)))
    plt.xticks([])
    plt.yticks([])
    
    img_path = "images/{}_fold/img2_{:02}.png".format(i_fold, i)
    plt.savefig(img_path)
    plt.close()
    # mlfr.log_artifact(img_path)

def plot_image_with_multiclasspred(X_data, Y_data, Y_pred, mlfr, i=0, key=None, threshold = 0.5, i_fold=1):

    if key is None:
        print("key is not set")

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(X_data)

    plt.subplot(2, 2, 2)
    plt.imshow(Y_data)

    plt.subplot(2, 2, 3)
    plt.imshow(X_data)

    plt.subplot(2, 2, 4)
    plt.imshow(mask2img(Y_pred, key))
    
    img_path = "images/{}_fold/img3_{:03}.png".format(i_fold, i)
    plt.savefig(img_path)
    plt.close()
    mlfr.log_artifact(img_path)

def resize9x16(img):
    rate = 120
    h = 9 * rate
    w = 16 * rate
    img = cv2.resize(img, (w, h))
    return img

def save_roc_curve(roc, i_fold=1):
    roc_x = roc[0]
    roc_y = roc[1]
    
    plt.figure(figsize=(7, 7))
    plt.plot(roc_x, roc_y)
    plt.savefig("images/{}_fold/roc_curve.png".format(i_fold))
    
    df_roc = pd.DataFrame({
        "roc_x": roc[0],
        "roc_y": roc[1],
    })
    df_roc.to_csv("images/{}_fold/roc_data.csv".format(i_fold), index=False)

def auc_using_roc_curve(roc):
    roc_x = roc[0]
    roc_y = roc[1]
    n = len(roc_x)
    auc = abs(sum(roc_x[i-1]*roc_y[i] - roc_x[i]*roc_y[i-1] for i in range(n))) / 2 + 0.5
    return auc