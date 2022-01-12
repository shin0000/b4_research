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

from .mytransforms import *


def make_used_data(data_dir, data):
    used_data = {k: [] for k in data}
    X_data_total_path_format = os.path.join(data_dir, "{}/movie")
    Y_data_total_path_format = os.path.join(data_dir, "{}/label")

    for d in data:
        data_numbers = []
        data_numbers1 = []
        data_numbers2 = []
        X_data_paths_candidate = X_data_total_path_format.format(d)
        Y_data_paths_candidate = Y_data_total_path_format.format(d)

        for data_path_candidate in os.listdir(X_data_paths_candidate):
            pattern = 'movieFrame_(\d+).png'
            number = re.match(pattern, data_path_candidate).group(1)
            data_numbers1.append(number)

        for data_path_candidate in os.listdir(Y_data_paths_candidate):
            pattern = 'label_(\d+).png'
            number = re.match(pattern, data_path_candidate).group(1)
            data_numbers2.append(number)

        data_numbers = list(set(data_numbers1) & set(data_numbers2))
        data_numbers = sorted(data_numbers)
        used_data[d] = data_numbers
    return used_data

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, used_data, SegClasses, used_organ, common_transform = None, inference=True, instruments_dir=None, instruments_transform=None):
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        if inference:
            self.target_transform = TargetToTensor()
        else:
            self.target_tramsform = TargetToNumpy()
        self.common_transform = common_transform
        self.used_data = used_data
        self.data = [k for k, v in used_data.items()]
        self.data_n = [len(v) for k, v in used_data.items()]
        self.data_cumn = np.cumsum(self.data_n)
        self.inference = inference
        self.SegClasses = SegClasses
        self.used_organ = used_organ
        self.instruments_dir = instruments_dir
        
        self.instruments_transform = instruments_transform
        if self.instruments_transform is None:
            self.instruments_transform = A.Compose([])
        if self.instruments_dir is not None:
            self.instruments_data = self.make_instruments_data(instruments_dir)
        else:
            self.instruments_data = None
    
    def __len__(self):
        return self.data_cumn[-1]
    
    def __getitem__(self, idx):
        ind = np.argmax(idx < self.data_cumn)
        d = self.data[ind]
        if ind != 0:
            idx = idx - self.data_cumn[ind-1]
        X_data_path, Y_data_path = self.return_data_path(d, idx)
        X_data, Y_data = self.return_data(X_data_path, Y_data_path)
        if not self.inference:
            return X_data, Y_data, Y_data_path
        return X_data, Y_data
    
    def return_data_path(self, d, i):
        number = self.used_data[d][i]

        X_data_format = "movieFrame_{}.png"
        Y_data_format = "label_{}.png"
        X_data_name = X_data_format.format(number)
        Y_data_name = Y_data_format.format(number)
        X_data_total_path = os.path.join(self.data_dir, "{}/movie".format(d))
        Y_data_total_path = os.path.join(self.data_dir, "{}/label".format(d))
        X_data_path = os.path.join(X_data_total_path, X_data_name)
        Y_data_path = os.path.join(Y_data_total_path, Y_data_name)

        return X_data_path, Y_data_path
    
    def return_data(self, X_data_path, Y_data_path):
        pil = False
        if pil:
            X_data = Image.open(X_data_path)
        else:
            X_data = cv2.imread(X_data_path)
            if X_data is None:
                print("X_data is None!")
            X_data = cv2.cvtColor(X_data, cv2.COLOR_BGR2RGB)
            
        if pil:
            Y_data = Image.open(Y_data_path)
        else:
            Y_data = cv2.imread(Y_data_path)
            Y_data = Y_data[..., 0]
            
        if self.inference:
            Y_data = self.decision_segclass_inference(Y_data)
        else:
            Y_data = self.decision_segclass_no_inference(Y_data)

        if self.instruments_data is not None:
            X_data, Y_data = self.instruments_over_img(self.instruments_dir, self.instruments_data, X_data, Y_data, transform=self.instruments_transform, imshow=False)
            
        if self.common_transform is not None:
            transformed = self.common_transform(image=X_data, mask=Y_data)
            X_data, Y_data = transformed["image"], transformed["mask"]

        # if self.instruments_data is not None:
        #     X_data, Y_data = self.instruments_over_img(self.instruments_dir, self.instruments_data, X_data, Y_data, transform=self.instruments_transform, imshow=False)
                
        if self.inference:
            X_data = self.transform(X_data)
            Y_data = self.target_transform(Y_data)
            
        return X_data, Y_data
    
    def decision_segclass_inference(self, Y_data):
        n_classes = len(self.used_organ)
        Y_data_seg = np.zeros((Y_data.shape[0], Y_data.shape[1], n_classes))
        for i, uo in enumerate(self.used_organ):
            label = self.SegClasses[uo]["label"]
            Y_data_seg[..., i] = np.where(Y_data == label, 1, 0)

        if n_classes == 1:
            return Y_data_seg[..., 0]
        else:
            print("Un Impletented")
            return Y_data_seg
    
    def decision_segclass_no_inference(self, Y_data):
        Y_data2 = np.zeros(Y_data.shape).astype(np.bool)
        Y_data2[...] = False
        for i, uo in enumerate(self.used_organ):
            label = self.SegClasses[uo]["label"]
            Y_data2 |= np.where(Y_data == label, True, False)
        
        Y_data2 = Y_data2.astype(np.uint8)
        return Y_data2

    def make_instruments_data(self, instruments_dir):
        instruments_movie_dir = os.path.join(instruments_dir, "movie")
        instruments_label_dir = os.path.join(instruments_dir, "label")

        instrument_movie_format = "movieFrame_{}.png"
        instrument_movie_pattern = r"movieFrame_([0-9]+).png"
        instrument_label_format = "label_{}.png"
        instrument_label_pattern = r"label_([0-9]+).png"

        numbers1 = []
        for instrument_movie_path in os.listdir(instruments_movie_dir):
            result = re.match(instrument_movie_pattern, instrument_movie_path)
            number = result.group(1)
            numbers1.append(number)

        numbers2 = []
        for instrument_label_path in os.listdir(instruments_label_dir):
            result = re.match(instrument_label_pattern, instrument_label_path)
            number = result.group(1)
            numbers2.append(number)

        numbers = list(set(numbers1) & set(numbers2))
        return numbers

    def instruments_over_img(self, instruments_dir, instruments_data, img_orig, mask_orig, transform=A.Compose([]), imshow=False):
        img = img_orig.copy()
        mask = mask_orig.copy()
        n = len(instruments_data)

        n_adding = np.random.randint(2)
        i = -1
        for j in range(n_adding):
            suggest_i = np.random.randint(n)
            if suggest_i != i:
                i = np.random.randint(n)
            else:
                continue

            number = instruments_data[i]
            instrument_movie_name = "movieFrame_{}.png".format(number)
            instrument_label_name = "label_{}.png".format(number)
            instrument_movie_path = os.path.join(os.path.join(instruments_dir, "movie"), instrument_movie_name)
            instrument_label_path = os.path.join(os.path.join(instruments_dir, "label"), instrument_label_name)

            img2 = cv2.imread(instrument_movie_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            mask2 = cv2.imread(instrument_label_path)
            mask2 = np.where(mask2 != 0, 1, 0)

            y_inds = np.where(mask2[..., 0] != 0)[0]
            x_inds = np.where(mask2[..., 0] != 0)[1]
            y_min_inds = np.min(y_inds)
            y_max_inds = np.max(y_inds)
            x_min_inds = np.min(x_inds)
            x_max_inds = np.max(x_inds)
            mask3 = mask2[y_min_inds: y_max_inds, x_min_inds: x_max_inds]
            img3 = img2[y_min_inds: y_max_inds, x_min_inds: x_max_inds]
            new_img3 = np.where(mask3 != 0, img3, 0)

            transformed = transform(image=img3, mask=mask3)
            img3, mask3 = transformed["image"].astype(np.uint8), transformed["mask"].astype(np.uint8)

            new_img3 = np.where(mask3 != 0, img3, 0)

            h = img.shape[0]
            w = img.shape[1]
            over_img = np.zeros(img.shape).astype(np.uint8)
            over_mask = np.zeros(img.shape).astype(np.uint8)
            
            r = np.random.randint(2)
            offset = 100
            if r == 0:
                img4 = np.rot90(img3)
                mask4 = np.rot90(mask3)
                if img4.shape[0] > h:
                    img4 = cv2.resize(img4, (h, img4.shape[1]))
                    mask4 = cv2.resize(mask4, (h, mask4.shape[1]))
                elif img4.shape[1] > w:
                    img4 = cv2.resize(img4, (img4.shape[0], w))
                    mask4 = cv2.resize(mask4, (mask4.shape[0], w))
                h_b = img4.shape[0]
                w_b = img4.shape[1]
                h_i = np.random.randint(h-h_b+1)
                w_i = w-w_b -offset
                over_img[h_i: h_i+h_b, w_i: w_i+w_b] = img4
                over_mask[h_i: h_i+h_b, w_i: w_i+w_b] = mask4
            else:
                if img3.shape[0] > h:
                    img3 = cv2.resize(img3, (h, img3.shape[1]))
                    mask3 = cv2.resize(mask3, (h, mask3.shape[1]))
                elif img3.shape[1] > w:
                    img3 = cv2.resize(img3, (img3.shape[0], w))
                    mask3 = cv2.resize(mask3, (mask3.shape[0], w))
                h_b = img3.shape[0]
                w_b = img3.shape[1]
                h_i = h-h_b
                w_i = np.random.randint(offset, w-w_b -offset)
                over_img[h_i: h_i+h_b, w_i: w_i+w_b] = img3
                over_mask[h_i: h_i+h_b, w_i: w_i+w_b] = mask3

            img[np.where(over_mask != 0)] = over_img[np.where(over_mask != 0)]
            mask[np.where(over_mask[..., 0] != 0)] = 0

        if imshow:
            new_img2 = (img2 * mask2).astype(np.uint8) # only instrument

            plt.figure(figsize=(15, 10))

            plt.subplot(3, 2, 1)
            plt.imshow(new_img2)

            plt.subplot(3, 2, 2)
            plt.imshow(mask2[..., 0])

            plt.subplot(3, 2, 3)
            plt.imshow(img_orig)

            plt.subplot(3, 2, 4)
            plt.imshow(mask_orig)

            plt.subplot(3, 2, 5)
            plt.imshow(img)

            plt.subplot(3, 2, 6)
            plt.imshow(mask)

            plt.show()

        return img, mask

def make_instruments_data(instruments_dir):
    instruments_movie_dir = os.path.join(instruments_dir, "movie")
    instruments_label_dir = os.path.join(instruments_dir, "label")

    instrument_movie_format = "movieFrame_{}.png"
    instrument_movie_pattern = r"movieFrame_([0-9]+).png"
    instrument_label_format = "label_{}.png"
    instrument_label_pattern = r"label_([0-9]+).png"

    numbers1 = []
    for instrument_movie_path in os.listdir(instruments_movie_dir):
        result = re.match(instrument_movie_pattern, instrument_movie_path)
        number = result.group(1)
        numbers1.append(number)

    numbers2 = []
    for instrument_label_path in os.listdir(instruments_label_dir):
        result = re.match(instrument_label_pattern, instrument_label_path)
        number = result.group(1)
        numbers2.append(number)

    numbers = list(set(numbers1) & set(numbers2))
    return numbers

def instruments_over_img(instruments_dir, instruments_data, img_orig, mask_orig, transform=A.Compose([])):
    img = img_orig.copy()
    mask = mask_orig.copy()
    n = len(instruments_data)

    n_adding = np.random.randint(2)
    n_adding = 1
    i = -1
    for j in range(n_adding):
        suggest_i = np.random.randint(n)
        if suggest_i != i:
            i = np.random.randint(n)
        # elif suggest_i in [1, 5, 10]:
        #     continue
        else:
            continue

        number = instruments_data[i]
        instrument_movie_name = "movieFrame_{}.png".format(number)
        instrument_label_name = "label_{}.png".format(number)
        instrument_movie_path = os.path.join(os.path.join(instruments_dir, "movie"), instrument_movie_name)
        instrument_label_path = os.path.join(os.path.join(instruments_dir, "label"), instrument_label_name)

        img2 = cv2.imread(instrument_movie_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        mask2 = cv2.imread(instrument_label_path)
        mask2 = np.where(mask2 != 0, 1, 0)

        y_inds = np.where(mask2[..., 0] != 0)[0]
        x_inds = np.where(mask2[..., 0] != 0)[1]
        y_min_inds = np.min(y_inds)
        y_max_inds = np.max(y_inds)
        x_min_inds = np.min(x_inds)
        x_max_inds = np.max(x_inds)
        mask3 = mask2[y_min_inds: y_max_inds, x_min_inds: x_max_inds]
        img3 = img2[y_min_inds: y_max_inds, x_min_inds: x_max_inds]
        new_img3 = np.where(mask3 != 0, img3, 0)

        transformed = transform(image=img3, mask=mask3)
        img3, mask3 = transformed["image"].astype(np.uint8), transformed["mask"].astype(np.uint8)

        new_img3 = np.where(mask3 != 0, img3, 0)

        h = img.shape[0]
        w = img.shape[1]
        over_img = np.zeros(img.shape).astype(np.uint8)
        over_mask = np.zeros(img.shape).astype(np.uint8)
        
        r = np.random.randint(2)
        offset = 100
        if r == 0:
            img4 = np.rot90(img3)
            mask4 = np.rot90(mask3)
            if img4.shape[0] > h:
                img4 = cv2.resize(img4, (h, img4.shape[1]))
                mask4 = cv2.resize(mask4, (h, mask4.shape[1]))
            elif img4.shape[1] > w:
                img4 = cv2.resize(img4, (img4.shape[0], w))
                mask4 = cv2.resize(mask4, (mask4.shape[0], w))
            h_b = img4.shape[0]
            w_b = img4.shape[1]
            h_i = np.random.randint(h-h_b+1)
            w_i = w-w_b -offset
            over_img[h_i: h_i+h_b, w_i: w_i+w_b] = img4
            over_mask[h_i: h_i+h_b, w_i: w_i+w_b] = mask4
        else:
            if img3.shape[0] > h:
                img3 = cv2.resize(img3, (h, img3.shape[1]))
                mask3 = cv2.resize(mask3, (h, mask3.shape[1]))
            elif img3.shape[1] > w:
                img3 = cv2.resize(img3, (img3.shape[0], w))
                mask3 = cv2.resize(mask3, (mask3.shape[0], w))
            h_b = img3.shape[0]
            w_b = img3.shape[1]
            h_i = h-h_b
            w_i = np.random.randint(offset, w-w_b -offset)
            over_img[h_i: h_i+h_b, w_i: w_i+w_b] = img3
            over_mask[h_i: h_i+h_b, w_i: w_i+w_b] = mask3

        img[np.where(over_mask != 0)] = over_img[np.where(over_mask != 0)]
        mask[np.where(over_mask[..., 0] != 0)] = 0

    return img, mask