#-*-coding:utf-8-*-

import os
import torch
try:
    from .utils import read_labels
    from .transform import Transform
    from .adjust_trainval import SampleAdjuster
except:
    from utils import read_labels
    from transform import Transform
    from adjust_trainval import SampleAdjuster


class ClassifyDataset:
    def __init__(self, img_path, label_cls, args):
        self.img_name, self.img_label = [], []
        self.label = label_cls
        self.label_nums = len(self.label)
        self.transform = Transform()
        self.transform.init_with_args(args)

        for idx, cls in enumerate(label_cls):
            for file in os.listdir(os.path.join(img_path, cls)):
                self.img_name.append(os.path.join(img_path, cls, file))
                self.img_label.append(idx)

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

        assert self.phases, "Please assign your phases using the dataset!"
        self.build(data_dir, label_path, inp_size, batch_size, num_worker)
        if adjust_ratio > 0 and self.phases is ("train", "val"):
            SampleAdjuster(data_dir, adjust_ratio).adjust_loader(self.dataloaders_dict)

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
    from config.train_args import args
    loader.build_with_args(args)
    vis_phase = "val"
    for idx, (name, inps, labels) in enumerate(loader.dataloaders_dict[vis_phase]):
        print(name)
