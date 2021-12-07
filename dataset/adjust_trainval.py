import os
import random
import shutil


class SampleAdjuster:
    def __init__(self, folder_path, ratio):
        self.folder_path = folder_path
        self.ratio = ratio

    def adjust(self, train_img, train_label, val_img, val_label):
        sample_sum = len(train_label) + len(val_label)
        val_sample_target = int(sample_sum * self.ratio)
        diff = len(val_label) - val_sample_target
        if diff < 0:
            diff_labels, diff_imgs, train_label, train_img = self.random_sample(diff, train_label, train_img)
            val_label += diff_labels
            val_img += diff_imgs
        else:
            diff_labels, diff_imgs, val_label, val_img = self.random_sample(diff, val_label, val_img)
            train_label += diff_labels
            train_img += diff_imgs
        return train_img, train_label, val_img, val_label

    def random_sample(self, num, labels, imgs):
        adjust_idx = random.sample(range(len(labels)), abs(num))
        adjust_labels, adjust_imgs, remaining_labels, remaining_imgs = [], [], [], []
        for idx, (label, img) in enumerate(zip(labels, imgs)):
            if idx in adjust_idx:
                adjust_labels.append(label)
                adjust_imgs.append(img)
            else:
                remaining_labels.append(label)
                remaining_imgs.append(img)
        return adjust_labels, adjust_imgs, remaining_labels, remaining_imgs

    def adjust_loader(self, loader):
        train_loader, val_loader = loader["train"], loader["val"]
        train_img, train_label = train_loader.img_name, train_loader.img_label
        val_img, val_label = val_loader.img_name, val_loader.img_label
        train_img, train_label, val_img, val_label = self.adjust(train_img, train_label, val_img, val_label)
        loader["train"].img_name, loader["train"].img_label = train_img, train_label
        loader["val"].img_name, loader["val"].img_label = val_img, val_label
        return loader


class ImgAdjuster:
    def __init__(self, val_r, src, mark="all"):
        self.val_ratio = val_r
        self.data_src = src
        if os.path.isdir("data"):
            self.train_src = os.path.join(self.data_src, "train")
            self.val_src = os.path.join(self.data_src, 'val')
        else:
            self.train_src = os.path.join("../data", self.data_src, "train")
            self.val_src = os.path.join("../data", self.data_src, 'val')
        self.type = os.listdir(self.train_src)
        self.train_path = ''
        self.val_path = ''
        self.train_ls = []
        self.val_ls = []
        self.class_mark = mark

    def adjust_img(self, class_type):
        self.train_path = os.path.join(self.train_src, class_type)
        self.val_path = os.path.join(self.val_src, class_type)
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)
        self.train_ls = os.listdir(self.train_path)
        self.val_ls = os.listdir(self.val_path)
        total_num = len(self.train_ls) + len(self.val_ls)
        new_val_num = total_num * self.val_ratio
        dis = int(new_val_num - len(self.val_ls))
        if dis > 0:
            move_ls = random.sample(self.train_ls, abs(dis))
            for pic in move_ls:
                shutil.move(os.path.join(self.train_path, pic), self.val_path)
        else:
            move_ls = random.sample(self.val_ls, abs(dis))
            for pic in move_ls:
                try:
                    shutil.move(os.path.join(self.val_path, pic), self.train_path)
                except shutil.Error:
                    pass

    def run(self):
        print("Adjusting validation proportion now...")
        for c in self.type:
            if self.class_mark == "all":
                self.adjust_img(c)
                print("All the samples in {0} have been adjusted to {1} val successfully".format(self.data_src,
                                                                                                 self.val_ratio))
            else:
                if c == self.class_mark:
                    self.adjust_img(c)
                    print("Sample {0} in {1} have been adjusted to {2} val successfully".format(c, self.data_src,
                                                                                                self.val_ratio))
                else:
                    pass