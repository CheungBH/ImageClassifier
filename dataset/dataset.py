#-*-coding:utf-8-*-

import os
import torch
try:
    from .utils import read_labels
    from .transform import Transform
except:
    from utils import read_labels
    from transform import Transform
from collections import Counter


class ClassifyDataset:
    def __init__(self, img_path, label_cls, args):
        img_dir_name, img_dir_label, self.img_name, self.img_label = [], [], [], []
        self.label = []
        image_label_dict = label_cls
        self.transform = Transform()
        self.transform.init_with_args(args)

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
        inp = self.transform.process(image_name)
        return image_name, inp, label


class DataLoader:
    def __init__(self, phases=("train", "val")):
        self.phases = phases

    def build_with_args(self, args, inp_size=224):
        adjust_ratio = args.trainval_ratio
        data_dir = args.data_path
        label_path = args.label_path
        batch_size = args.batch_size
        num_worker = args.num_worker

        if adjust_ratio > 0 and self.phases is ("train", "val"):
            ImgAdjuster(adjust_ratio, data_dir).run()
        assert self.phases, "Please assign your phases using the dataset!"
        self.build(data_dir, label_path, inp_size, batch_size, num_worker)

    def build(self, data_dir, label_path="", inp_size=224, batch_size=32, num_worker=2, shuffle=True):
        self.label = self.get_labels(data_dir, label_path)
        self.image_datasets = {x: ClassifyDataset(os.path.join(data_dir, x), self.label, size=inp_size) for x in
                               self.phases}
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                                                shuffle=shuffle, num_workers=num_worker)
                                 for x in self.phases}
        self.cls_num = len(self.image_datasets[self.phases[0]].label)

    def get_labels(self, img_dir, label_path):
        if label_path:
            return read_labels(label_path)
        label_path = os.path.join(img_dir, "labels.txt")
        if os.path.exists(label_path):
            return read_labels(label_path)
        else:
            phase_dir = os.path.join(img_dir, "train")
            labels = [cls for cls in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, cls))]
            with open(label_path, "w") as f:
                for label in labels:
                    f.write(label + "\n")
            return labels


if __name__ == '__main__':
    data_dir = "/home/hkuit155/Desktop/CNN_classification/data/CatDog"
    loader = DataLoader()
    loader.build(data_dir)
    vis_phase = "val"
    for idx, (name, inps, labels) in enumerate(loader.dataloaders_dict[vis_phase]):
        print(name)
