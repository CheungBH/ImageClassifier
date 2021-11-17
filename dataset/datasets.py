from __future__ import print_function
import torch
from torch.utils.data import Dataset
from .utils import image_normalize
import numpy as np
from tool.adjust_val import ImgAdjuster
import os
import cv2
from collections import Counter

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]

class ClassificationDatasets(Dataset):
    def __init__(self, img_path, image_label_dict, phase="train", size=224):
        img_dir_name, img_dir_label, self.img_name, self.img_label = [], [], [], []
        self.size = size
        self.label = []

        for idx, cls in enumerate(image_label_dict):
            self.label.append(cls)
            img_dir_name.append(os.path.join(img_path, cls))
            img_dir_label.append(idx)

        for dir_name, dir_label in zip(img_dir_name, img_dir_label):
            img_file_names = os.listdir(os.path.join(dir_name))
            for img_name in img_file_names:
                self.img_name.append(os.path.join(dir_name, img_name))
                self.img_label.append(dir_label)

        self.label_nums = Counter(self.img_label)

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        image_name = self.img_name[item]
        label = self.img_label[item]
        try:
            image_object = self.image_preprocess(image_name, size=self.size)
            return image_name, image_object, label
        except Exception as e:
            print("error read ", image_name, e)
            # os.remove(image_name)
            image_name = self.img_name[0]
            _label = self.img_label[0]
            _image_object = self.image_preprocess(image_name, size=self.size)
            return image_name, _image_object, _label

    def image_preprocess(self, img_name, size=224):
        if isinstance(img_name, str):
            image_array = cv2.imread(img_name)
        else:
            image_array = img_name
        image_array = cv2.resize(image_array, (size, size))
        image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
        image_array = image_array.transpose((2, 0, 1))
        for channel, _ in enumerate(image_array):
            image_array[channel] /= 255.0
            image_array[channel] -= image_normalize_mean[channel]
            image_array[channel] /= image_normalize_std[channel]
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor




