import numpy as np
import torch
import random
import os
import cv2

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def disentangleKey(key):
    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        category = key[i]['name']
        super_category = key[i]['super-category']
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0,c1,c2])
        dKey[class_id] = {'color': color_array, 'name': category, 'super_category': super_category}

    return dKey

def resize9x16(img):
    rate = 120
    h = 9 * rate
    w = 16 * rate
    img = cv2.resize(img, (w, h))
    return img