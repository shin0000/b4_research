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

def train_model(model, criterion, metricfn, optimizer, scheduler, train_dataloader, valid_dataloader, epochs, device, mlfr, patience=7, save_model_path="models_weights/model.pth", i_fold=1, n_inference=1):
    model = model.to(device)
    p = 0
    best_loss = np.inf
    best_metric = 0
    for epoch in range(epochs):
        print("{} / {} epoch ".format(epoch+1, epochs))
        train_losses = []
        train_metrics = []
        valid_losses = []
        valid_metrics = []
        tmp = []

        model.train()
        for X_train, Y_train in train_dataloader:
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            Y_train_pred_stack = []
            with torch.no_grad():
                for j in range(n_inference):
                    Y_train_pred = model(X_train)
                    Y_train_pred = Y_train_pred.sigmoid()
                    Y_train_pred_stack.append(Y_train_pred)
            Y_uncertainty = Y_preds2entropy(Y_train_pred_stack, device)
            Y_uncertainty_max = Y_preds2entropymax(Y_train_pred_stack, device)
            # Y_uncertainty_std = Y_preds2std(Y_train_pred_stack, device)
            tmp.append(Y_uncertainty.mean().item())
            optimizer.zero_grad()
            Y_train_pred = model(X_train)
            loss = criterion(Y_train_pred, Y_train, Y_uncertainty_max)
            loss.backward()
            optimizer.step()
            metric = metricfn(Y_train_pred, Y_train)
            train_losses.append(loss.item())
            train_metrics.append(metric.item())
        print("train: loss {}, metric {}".format(np.mean(train_losses), np.mean(train_metrics)))
        mlfr.log_metric("Entropy {} fold".format(i_fold), np.mean(tmp), step=epoch+1)

        model.eval()
        if n_inference > 1:
            if str(model.__class__) == "<class 'torch.nn.parallel.data_parallel.DataParallel'>":
                model.module.enable_dropout()
            else:
                model.enable_dropout()
        with torch.no_grad():
            for X_valid, Y_valid in valid_dataloader:
                X_valid, Y_valid = X_valid.to(device), Y_valid.to(device)
                Y_valid_preds_n = []
                for j in range(n_inference):
                    Y_valid_pred = model(X_valid)
                    Y_valid_preds_n.append(Y_valid_pred)
                Y_valid_pred = torch.stack(Y_valid_preds_n).mean(axis=0)
                loss = criterion(Y_valid_pred, Y_valid)
                metric = metricfn(Y_valid_pred, Y_valid)
                valid_losses.append(loss.item())
                valid_metrics.append(metric.item())
            scheduler.step(np.mean(valid_losses))
            print("valid: loss {}, metric {}".format(np.mean(valid_losses), np.mean(valid_metrics)))

        mlfr.log_metric("Train Loss {} fold".format(i_fold), np.mean(train_losses), step=epoch+1)
        mlfr.log_metric("Train Metrics {} fold".format(i_fold), np.mean(train_metrics), step=epoch+1)
        mlfr.log_metric("Valid Loss {} fold".format(i_fold), np.mean(valid_losses), step=epoch+1)
        mlfr.log_metric("Valid Metrics {} fold".format(i_fold), np.mean(valid_metrics), step=epoch+1)

        if np.mean(valid_metrics) > best_metric:
            p = 0
            best_loss = np.mean(valid_losses)
            best_metric = np.mean(valid_metrics)
            torch.save(model.state_dict(), save_model_path.format(i_fold))
        else:
            p += 1

        if p > patience:
            print("-- EarlyStopping --")
            break

def Y_preds2entropy(Y_pred_stack, device):
    Y_pred_stack = torch.stack(Y_pred_stack).to(device)
    Y_pred_stack = torch.where(Y_pred_stack < 1e-4, torch.tensor(1e-4, dtype=Y_pred_stack.dtype).to(device), Y_pred_stack)
    Y_pred_stack = torch.where(Y_pred_stack > 1 - 1e-4, torch.tensor(1 - 1e-4, dtype=Y_pred_stack.dtype).to(device), Y_pred_stack)
    Y_pred_stack = - Y_pred_stack * torch.log2(Y_pred_stack) - (1 - Y_pred_stack) * torch.log2(1 - Y_pred_stack)
    Y_pred_stack = Y_pred_stack.mean(axis=0)
    return Y_pred_stack

def Y_preds2entropymax(Y_pred_stack, device):
    Y_pred_stack = torch.stack(Y_pred_stack).to(device)
    Y_pred_stack = torch.where(Y_pred_stack < 1e-4, torch.tensor(1e-4, dtype=Y_pred_stack.dtype).to(device), Y_pred_stack)
    Y_pred_stack = torch.where(Y_pred_stack > 1 - 1e-4, torch.tensor(1 - 1e-4, dtype=Y_pred_stack.dtype).to(device), Y_pred_stack)
    Y_pred_stack = - Y_pred_stack * torch.log2(Y_pred_stack) - (1 - Y_pred_stack) * torch.log2(1 - Y_pred_stack)
    Y_pred_stack = Y_pred_stack.max(axis=0).values
    return Y_pred_stack

def Y_preds2std(Y_pred_stack, device):
    Y_pred_stack = torch.stack(Y_pred_stack).to(device)
    Y_pred_stack = Y_pred_stack.std(axis=0)
    Y_pred_stack_min = Y_pred_stack.min()
    Y_pred_stack_max = Y_pred_stack.max()
    Y_pred_stack = (Y_pred_stack - Y_pred_stack_min) / (Y_pred_stack_max - Y_pred_stack_min)
    Y_pred_stack = Y_pred_stack + 1.0
    return Y_pred_stack