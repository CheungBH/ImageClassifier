from __future__ import print_function

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

def image_normalize(image_array):
    image_normalize_mean = [0.485, 0.456, 0.406]
    image_normalize_std = [0.229, 0.224, 0.225]
    image_array = cv2.resize(image_array, (224, 224))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)  #返回和传入数组类似的内存中连续的数组
    image_array = image_array.transpose((2, 0, 1))
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image,image = get_image_tensor_list(raw_image)
    if raw_image is not None:
        raw_image = cv2.resize(raw_image, (224,) * 2)
    return image, raw_image

def get_image_tensor_list(img):
    img_tensor_list = []
    img_tensor = image_normalize(img)
    img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
    return img, img_tensor
