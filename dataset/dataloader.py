from __future__ import print_function
import torch
from torch.utils.data import Dataset
from .utils import image_normalize, read_labels
from .adjust_trainval import ImgAdjuster
import os
from collections import Counter


class ClassifyDataset(Dataset):
    def __init__(self, img_path, label_cls, image_processor=image_normalize, size=224):
        img_dir_name, img_dir_label, self.img_name, self.img_label = [], [], [], []
        self.size = size
        self.label = []
        image_label_dict = label_cls
        # image_label_dict = os.listdir(img_path)

        for idx, cls in enumerate(image_label_dict):
            self.label.append(cls)
            img_dir_name.append(os.path.join(img_path, cls))
            img_dir_label.append(idx)

        for dir_name, dir_label in zip(img_dir_name, img_dir_label):
            img_file_names = os.listdir(os.path.join(dir_name))
            for img_name in img_file_names:
                self.img_name.append(os.path.join(dir_name, img_name))
                self.img_label.append(dir_label)

        self.image_processor = image_processor
        self.label_nums = Counter(self.img_label)

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        image_name = self.img_name[item]
        label = self.img_label[item]
        try:
            image_object = self.image_processor(image_name, size=self.size)
            return image_name, image_object, label
        except Exception as e:
            print("error read ", image_name, e)
            # os.remove(image_name)
            image_name = self.img_name[0]
            _label = self.img_label[0]
            _image_object = self.image_processor(image_name, size=self.size)
            return image_name, _image_object, _label

    # def get_labels(self, img_dir, label_path):
    #     if label_path:
    #         return read_labels(label_path)
    #     label_path = os.path.join(img_dir, "labels.txt")
    #     if os.path.exists(label_path):
    #         return read_labels(label_path)
    #     else:
    #         labels = [cls for cls in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, cls))]
    #         with open(label_path, "w") as f:
    #             for label in labels:
    #                 f.write(label + "\n")
    #         return labels


class DataLoader(object):
    def __init__(self, data_dir, phases=("train", "val"), label_path="", batch_size=8, num_worker=2, inp_size=224, adjust_ratio=-1):
        if adjust_ratio > 0 and phases is ("train", "val"):
            ImgAdjuster(adjust_ratio, data_dir).run()
        assert phases, "Please assign your phases using the dataset!"
        self.label = self.get_labels(data_dir, label_path)
        self.image_datasets = {x: ClassifyDataset(os.path.join(data_dir, x), self.label, size=inp_size) for x in phases}
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                                                shuffle=True, num_workers=num_worker)
                            for x in phases}
        self.cls_num = len(self.image_datasets[phases[0]].label)

    def get_labels(self, img_dir, label_path):
        if label_path:
            return read_labels(label_path)
        label_path = os.path.join(img_dir, "labels.txt")
        if os.path.exists(label_path):
            return read_labels(label_path)
        else:
            phase_dir = os.path.join(img_dir, os.listdir(img_dir)[0])
            labels = [cls for cls in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, cls))]
            with open(label_path, "w") as f:
                for label in labels:
                    f.write(label + "\n")
            return labels
