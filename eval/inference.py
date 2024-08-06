#-*-coding:utf-8-*-
from dataset.utils import image_normalize, read_labels, get_pretrain
from models.build import ModelBuilder
import cv2
import config.config as config
import random
import torch


class ModelInference:
    def __init__(self, model_path, label_path, backbone, visualize, inp_size, device="cuda:0"):
        self.backbone = backbone if backbone else get_pretrain(model_path)
        self.model_size = inp_size
        self.classes = read_labels(label_path)
        self.MB = ModelBuilder()
        self.model = self.MB.build(len(self.classes), self.backbone, device)
        self.MB.load_weight(model_path)
        self.model.eval()
        self.visualize = visualize
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

    def run(self, img, cnt=0):
        img_tns = image_normalize(img, size=self.model_size)
        self.scores = self.MB.inference(img_tns)
        # self.exists = self.scores > 0.5
        # _, self.pred_idx = torch.max(self.scores, 1)
        # self.pred_cls = [cls for cls, exist in zip(self.classes, self.exists) if exist]
        font = cv2.FONT_HERSHEY_SIMPLEX
        # if self.visualize:
        cls_cnt = 0
        for i, score in enumerate(self.scores[0]):
            if score < 0.5:
                continue
            string = "{}: {:.2f}".format(self.classes[i], score)
            img = cv2.putText(img, string, (50, 50 + 50 * i), font, 2, self.colors[i], 3)
            cls_cnt += 1
        # cv2.putText(img, string, (50, 50), font, 2, self.colors[self.pred_idx], 3)
        return img

    def release(self):
        return
