import numpy as np
import matplotlib.pyplot as plt

def plot_image(X_data, Y_data, plot=False):
    plt.figure(figsize=(20, 30))
    plt.subplot(1, 3, 1)
    plt.imshow(X_data)

    plt.subplot(1, 3, 2)
    plt.imshow(Y_data)

    plt.subplot(1, 3, 3)
    Y_data = np.array(Y_data)
    if len(Y_data.shape) != 3:
        mask = np.zeros((*Y_data.shape, 3))
        mask[:, :, 0], mask[:, :, 1], mask[:, :, 2] = Y_data, Y_data, Y_data
        plt.imshow(np.where(mask != 0, X_data, 0))
    else:
        plt.imshow(np.where(Y_data != 0, X_data, 0))

    if plot:
        plt.show()
    else:
        plt.savefig("images/img1.png")
        plt.close()

def plot_instruments_over_img(new_img2, mask2, img_orig, mask_orig, img, mask):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.imshow(new_img2) #オーグメンテーション用画像:マスクとRGB画像の重ね合わせ

    plt.subplot(3, 2, 2)
    plt.imshow(mask2[..., 0]) #オーグメンテーション用画像:マスク

    plt.subplot(3, 2, 3)
    plt.imshow(img_orig) #オリジナル: RGB画像

    plt.subplot(3, 2, 4)
    plt.imshow(mask_orig) #オリジナル:マスク

    plt.subplot(3, 2, 5)
    plt.imshow(img) #合成: RGB画像

    plt.subplot(3, 2, 6)
    plt.imshow(mask) #合成: マスク

    plt.show()

def mask2img(mask, key):
    pred_mask = np.argmax(mask, axis=0)
    pred_img = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    for k, v in key.items():
        pred_img[np.where(pred_mask==k)] = v["color"]
    pred_img = pred_img.astype(np.uint8)
    return pred_img

def resize9x16(img):
    rate = 120
    h = 9 * rate
    w = 16 * rate
    img = cv2.resize(img, (w, h))
    return img