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

from utils import *
from implementations import *

mlfr1 = MLFlowRecord()
mlfr2 = MLFlowRecordDummy()

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    print("pre train: ", cfg.model_info.pre_train)
    print("MLFlow used: ", cfg.mlfr_used)
    print("used device: ", cfg.gpu.device)
    print("multi_gpu: ", cfg.gpu.multi_gpu)

    try:
        os.mkdir("models_weights")
        os.mkdir("images")
    except:
        pass

    current_dir = hydra.utils.get_original_cwd()
    data_dir = os.path.join(current_dir, "../data_NuVAT/pancreas")
    
    if cfg.augmentation.data_aug :
        train_transform = train_transform1
        # augmentation_mode = 2
    else:
        train_transform = train_transform2
        # augmentation_mode = 1

    if cfg.mlfr_used:
        mlfr = mlfr1
    else:
        mlfr = mlfr2

    if  cfg.augmentation.instruments_aug:
        instruments_dir = "../data_NuVAT/pancreas/{}".format(cfg.augmentation.aug_dir)
        instruments_dir = os.path.join(current_dir, instruments_dir)
        instruments_transform = instruments_transform1
        # augmentation_mode = 3
    elif  cfg.augmentation.cutmix_aug:
        instruments_dir = "../data_NuVAT/pancreas/{}".format(cfg.augmentation.aug_dir)
        instruments_dir = os.path.join(current_dir, instruments_dir)
        instruments_transform = instruments_transform1
        # augmentation_mode = 4
    else:
        instruments_dir = None
        instruments_transform = None

    data = ["Case001-1", "Case001-2", "Case002-1", "Case003-2", "Case004-1", "Case006-1", "Case007-1", "Case008-1", "Case008-2", "Case009-2", "Case010-2"]

    SegClassesPath = "./SegClasses.json"
    SegClassesPath = os.path.join(current_dir, SegClassesPath)
    SegClasses = json.load(open(SegClassesPath, "r"))
    used_organ = ["pancreas"]

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

    # experiment_name = "{}_augmentation{}".format(cfg.model_info.model, augmentation_mode)
    experiment_name = "{}-{}".format(cfg.model_info.name, cfg.augmentation.name)
    mlfr.set_experiment(experiment_name)
    mlfr.start_run()

    mlfr.log_param("a. data augmentation", cfg.augmentation.data_aug)
    mlfr.log_param("a. instruments augmentation", cfg.augmentation.instruments_aug)
    mlfr.log_param("a. cutmix augmentation", cfg.augmentation.cutmix_aug)

    mlfr.log_param("b. batch size", cfg.model_info.batch_size)
    mlfr.log_param("b. model", cfg.model_info.model)
    mlfr.log_param("b. optimizer", cfg.model_info.optimizer)
    mlfr.log_param("b. loss", cfg.model_info.loss)
    mlfr.log_param("b. metric", cfg.model_info.metric)

    mlfr.set_tag("comment", cfg.comment)
    mlfr.set_tag("seed", cfg.seed)
    mlfr.set_tag("cross_validation", cfg.cross_validation)

    if cfg.cross_validation:
        mlfr.set_tag("n_fold", len(folds))

    folds_metrics = []
    folds_losses = []
    folds_f = []
    folds_precision = []
    folds_recall = []
    folds_auc = []
    for i in range(len(folds)):
        i_fold = i + 1
        if cfg.cross_validation:
            print("{}/ {} fold".format(i_fold, len(folds)))
        else:
            if i > 0:
                break

        try:
            os.mkdir("models_weights/{}_fold".format(i_fold))
            os.mkdir("images/{}_fold".format(i_fold))
        except:
            pass

        fold_data = folds[i]
        train_data = fold_data["train_data"]
        valid_data = fold_data["valid_data"]
        test_data = fold_data["test_data"]

        print("train data: {}".format(train_data))
        print("valid data: {}".format(valid_data))
        print("test data: {}".format(test_data))

        train_used_data = make_used_data(data_dir, train_data)
        valid_used_data = make_used_data(data_dir, valid_data)
        test_used_data = make_used_data(data_dir, test_data)

        train_dataset = CustomDataset(data_dir, train_used_data, SegClasses, used_organ, common_transform=train_transform, instruments_dir=instruments_dir, instruments_transform=instruments_transform, inference=True)
        valid_dataset = CustomDataset(data_dir, valid_used_data, SegClasses, used_organ, common_transform=evaluate_transform, inference=True)
        test_dataset = CustomDataset(data_dir, test_used_data, SegClasses, used_organ, common_transform=evaluate_transform, inference=True)
        test_dataset_orig = CustomDataset(data_dir, test_used_data, SegClasses, used_organ, common_transform=evaluate_transform, inference=False)

        print("n train data {} fold".format(i_fold), len(train_dataset))
        print("n valid data {} fold".format(i_fold), len(valid_dataset))
        print("n test data {} fold".format(i_fold), len(test_dataset))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model_info.batch_size, shuffle=True, num_workers=8)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.model_info.batch_size, shuffle=False, num_workers=8)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.model_info.batch_size, shuffle=False, num_workers=8)

        # Train phase
        if cfg.model_info.model == "BayesianUNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "UNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "AttentionUNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "BayesianAttentionUNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "DeepLabV3":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels, cfg.model_info.backbone)
        else:
            print("未実装")

        mlfr.set_tag("n train data {} fold".format(i_fold), len(train_dataset))
        mlfr.set_tag("n valid data {} fold".format(i_fold), len(valid_dataset))
        mlfr.set_tag("n test data {} fold".format(i_fold), len(test_dataset))

        criterion = Criterion[cfg.model_info.loss]()
        optimizer = Optimizer[cfg.model_info.optimizer](model.parameters(), lr=cfg.model_info.learning_rate)
        metricfn = Metric[cfg.model_info.metric]   
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=cfg.model_info.scheduler_patience)

        if (cfg.gpu.device[:4] == "cuda") & (cfg.gpu.multi_gpu):
            model = nn.DataParallel(model, device_ids=cfg.gpu.multi_device_ids)

        train_model(model, criterion, metricfn, optimizer, scheduler, train_dataloader, valid_dataloader, cfg.model_info.epochs, cfg.gpu.device, mlfr, cfg.model_info.patience, save_model_path=cfg.model_info.save_model_path, i_fold=i_fold, n_inference=cfg.model_info.n_inference)

        # Evaluate phase
        if cfg.model_info.model == "BayesianUNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "UNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "AttentionUNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "BayesianAttentionUNet":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels)
        elif cfg.model_info.model == "DeepLabV3":
            model = Model[cfg.model_info.model](cfg.model_info.in_channels, cfg.model_info.out_channels, cfg.model_info.backbone)
        else:
            print("未実装")

        model = model.to(cfg.gpu.device)
        if (cfg.gpu.device[:4] == "cuda") & (cfg.gpu.multi_gpu):
            model = nn.DataParallel(model, device_ids=cfg.gpu.multi_device_ids)

        model.load_state_dict(torch.load(cfg.model_info.save_model_path.format(i_fold)))

        test_loss, test_metric, test_f, test_precision, test_recall, test_auc = evaluate_model(model, criterion, metricfn, test_dataloader, cfg.gpu.device, mlfr, i_fold, n_inference=cfg.model_info.n_inference)
        folds_f.append(test_f)
        folds_precision.append(test_precision)
        folds_recall.append(test_recall)
        folds_auc.append(test_auc)
        folds_metrics.append(test_metric)
        folds_losses.append(test_loss)

        for i in range(len(test_dataset)):
            compare_pred_label(model, test_dataset, test_dataset_orig, cfg.gpu.device, mlfr, i=i, i_fold=i_fold, n_inference=cfg.model_info.n_inference)
            if i+1 > 30:
                break

        mlfr.log_artifact("models_weights/{}_fold".format(i_fold))
        mlfr.log_artifact("images/{}_fold".format(i_fold))

    print("final score: loss {}, metric {}".format(np.mean(folds_losses), np.mean(folds_metrics)))
    mlfr.log_metric("Test Loss ", np.mean(folds_losses))
    mlfr.log_metric("Test Metrics ", np.mean(folds_metrics))
    mlfr.log_metric("Test F", np.mean(folds_f))
    mlfr.log_metric("Test Precision", np.mean(folds_precision))
    mlfr.log_metric("Test Recall", np.mean(folds_recall))
    mlfr.log_metric("Test AUC", np.mean(folds_auc))
    shutil.rmtree("models_weights/")
    shutil.rmtree("images/")

    mlfr.end_run()

if __name__ == "__main__":
    main()
