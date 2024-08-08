from __future__ import print_function
import torch
from torch.utils.data import Dataset
from .utils import image_normalize, read_labels
from .adjust_trainval import ImgAdjuster
import os
from PIL import Image
from collections import Counter
from torchvision import transforms
import random


class ClassifyDataset(Dataset):
    def __init__(self, img_path, label_cls, settings, is_train=True, data_percentage=1):
        img_dir_name, img_dir_label, self.img_name, self.img_label = [], [], [], []
        self.data_percentage = data_percentage
        self.size = settings["model"]["input_size"]
        self.label = []
        self.is_train = is_train
        self.init_augment(settings)
        image_label_dict = label_cls
        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings["transform"]["mean"], std=settings["transform"]["std"])
        ])
        # image_label_dict = os.listdir(img_path)

        for idx, cls in enumerate(image_label_dict):
            self.label.append(cls)
            img_dir_name.append(os.path.join(img_path, cls))
            img_dir_label.append(idx)

        for dir_name, dir_label in zip(img_dir_name, img_dir_label):
            img_file_names = os.listdir(os.path.join(dir_name))
            for idx, img_name in enumerate(img_file_names):
                if (idx * self.data_percentage) % 1 != 0:
                    continue
                self.img_name.append(os.path.join(dir_name, img_name))
                self.img_label.append(dir_label)

        # self.image_processor = image_processor
        self.label_nums = Counter(self.img_label)

    def __len__(self):
        return len(self.img_name)

    def init_augment(self, settings):
        # https://www.cnblogs.com/jgg54335/p/14572640.html
        self.flip_ratio = settings["transform"]["flip"]["p"]
        self.rotate_max = settings["transform"]["rotate"]["max_factor"]
        self.rotate_ratio = settings["transform"]["rotate"]["p"]
        self.contrast_ratio = settings["transform"]["contrast"]["p"]
        self.contrast_max = settings["transform"]["contrast"]["max_factor"]
        self.brightness_ratio = settings["transform"]["brightness"]["p"]
        self.brightness_max = settings["transform"]["brightness"]["max_factor"]

    def __getitem__(self, item):
        image_name = self.img_name[item]
        label = self.img_label[item]
        image = Image.open(image_name).convert("RGB")
        if self.is_train:
            image = self.augment(image)
        image = self.transforms(image)
        return image_name, image, label

    def augment(self, image):
        image = transforms.RandomHorizontalFlip(p=self.flip_ratio)(image)
        if random.random() < self.rotate_ratio:
            image = transforms.RandomRotation([-self.rotate_max, self.rotate_max])(image)
        if random.random() < self.contrast_ratio:
            image = transforms.ColorJitter(contrast=self.contrast_max)(image)
        if random.random() < self.brightness_ratio:
            # rand_brightness = random.uniform(1 - self.brightness_max, 1 + self.brightness_max)
            image = transforms.ColorJitter(brightness=self.brightness_max)(image)
        return image



class DataLoader:
    def __init__(self,  phases=("train", "val")):
        self.phases = phases

    def build_with_args(self, args, settings):
        data_percentage = args.data_percentage
        adjust_ratio = args.trainval_ratio
        data_dir = args.data_path
        label_path = args.label_path
        batch_size = args.batch_size
        num_worker = args.num_worker

        if adjust_ratio > 0 and self.phases is ("train", "val"):
            ImgAdjuster(adjust_ratio, data_dir).run()
        assert self.phases, "Please assign your phases using the dataset!"
        self.build(data_dir, settings, label_path, batch_size, num_worker, data_percentage=data_percentage)

    def build(self, data_dir, settings, label_path="", batch_size=32, num_worker=2, shuffle=True, **kwargs):
        self.label, self.label_path = self.get_labels(data_dir, label_path)
        self.image_datasets = {x: ClassifyDataset(os.path.join(data_dir, x), self.label, is_train=x == "train", settings=settings, **kwargs) for x in self.phases}
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                                                shuffle=shuffle, num_workers=num_worker)
                            for x in self.phases}
        self.cls_num = len(self.image_datasets[self.phases[0]].label)

    def get_labels(self, img_dir, label_path):
        if label_path:
            return read_labels(label_path)
        label_path = os.path.join(img_dir, "labels.txt")
        if os.path.exists(label_path):
            return read_labels(label_path), label_path
        else:
            phase_dir = os.path.join(img_dir, "train")
            labels = [cls for cls in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, cls))]
            with open(label_path, "w") as f:
                for label in labels:
                    f.write(label + "\n")
            return labels, label_path

