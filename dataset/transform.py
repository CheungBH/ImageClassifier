#-*-coding:utf-8-*-
import cv2
from torchvision.transforms import functional as F
import random
import math
import numpy as np


class Transform:
    def __init__(self, train=True):
        self.color = "rgb"
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.img_aug = True if train else False

    def init_with_args(self, args):
        self.flip_prob = args.flip_prob
        self.rotate_prob = args.rotate_prob
        self.max_rotate_angle = args.rotate_angle

    def init(self, flip_prob, rotate_prob, rotate_max_angle):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotate_angle = rotate_max_angle

    def load_img(self, img_path):
        try:
            img = cv2.imread(img_path)
            if self.color == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            import sys
            print(img_path)
            sys.exit()
        return img

    def convert(self, img, src="bgr", dest="rgb"):
        if src == "bgr" and dest == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def img2tensor(self, img):
        return F.to_tensor(img)

    def normalize(self, img):
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def flip(self, img):
        return cv2.flip(img, 1)

    def rotate(self, img, degree):
        img_h, img_w = img.shape[0], img.shape[1]
        center_img = (img_w/2, img_h/2)
        R_img = cv2.getRotationMatrix2D(center_img, degree, 1)
        cos, sin = abs(R_img[0, 0]), abs(R_img[0, 1])
        new_img_w = int(img_w * cos + img_h * sin)
        new_img_h = int(img_w * sin + img_h * cos)
        new_img_size = (new_img_w, new_img_h)
        R_img[0, 2] += new_img_w / 2 - center_img[0]
        R_img[1, 2] += new_img_h / 2 - center_img[1]
        img_new = cv2.warpAffine(img, R_img, dsize=new_img_size,borderMode=cv2.BORDER_CONSTANT)
        return img_new

    def process(self, img_path):
        raw_img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
        img = self.convert(raw_img, "bgr", self.color)
        if self.img_aug:
            if random.random() > 1 - self.flip_prob:
                img = self.flip(img)
            if random.random() > 1 - self.rotate_prob:
                degree = (random.random() - 0.5) * 2 * self.max_rotate_angle
                img = self.rotate(img, degree)
        inp = self.normalize(self.img2tensor(img))
        return inp


if __name__ == '__main__':
    img_p = "../data/cat_dog_test/12413.jpg"
    t = Transform()
    img = t.rotate(cv2.imread(img_p), 40)
    cv2.imshow("img", img)
    cv2.waitKey(0)


