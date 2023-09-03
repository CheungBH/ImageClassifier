#-*-coding:utf-8-*-
from dataset.utils import image_normalize, read_labels, get_pretrain
from models.build import ModelBuilder
import cv2
import config.config as config
import random
import torch


class ModelInference:
    def __init__(self, model_path, label_path, backbone, visualize, device="cuda:0"):
        self.backbone = backbone if backbone else get_pretrain(model_path)
        self.model_size = 224
        self.classes = read_labels(label_path)
        self.MB = ModelBuilder()
        self.model = self.MB.build(len(self.classes), self.backbone, device)
        self.MB.load_weight(model_path)
        self.model.eval()
        self.visualize = visualize
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

    def run(self, img, cnt=0):
        img_tns = image_normalize(img, size=224)
        self.scores = self.MB.inference(img_tns)
        _, self.pred_idx = torch.max(self.scores, 1)
        self.pred_cls = self.classes[self.pred_idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.visualize:
            cv2.putText(img, self.pred_cls, (50, 50), font, 2, self.colors[self.pred_idx], 3)
        return img

    def release(self):
        return
