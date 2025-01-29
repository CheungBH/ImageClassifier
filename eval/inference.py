#-*-coding:utf-8-*-
from dataset.utils import image_normalize, read_labels, get_pretrain
from models.build import ModelBuilder
import cv2
import random
import torch


class ModelInference:
    def __init__(self, model_path, label_path, backbone, visualize, inp_size, device="cuda:0"):
        self.backbone = backbone if backbone else get_pretrain(model_path)
        self.model_size = inp_size
        self.classes = read_labels(label_path)
        self.confidence_thresh = [-1 for _ in range(len(self.classes))]
        self.MB = ModelBuilder()
        self.model = self.MB.build(len(self.classes), self.backbone, device)
        self.MB.load_weight(model_path)
        self.model.eval()
        self.visualize = visualize
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        self.confidence_thresh = None
        # self.confidence_thresh = {1: 0.8, 2: 0.9}
        self.decimal = 4

    def get_pred(self):
        if self.confidence_thresh is None:
            return self.max_cls, self.scores[self.max_idx]
        for cls_idx, thresh in self.confidence_thresh.items():
            if self.scores[cls_idx] > thresh:
                return self.classes[cls_idx], self.scores[cls_idx]
        return self.max_cls, self.scores[self.max_idx]

    def run(self, img, cnt=0):
        img_tns = image_normalize(img, size=self.model_size)
        scores = self.MB.inference(img_tns)
        _, self.max_idx = torch.max(scores, 1)
        self.scores = scores[0]
        self.max_cls = self.classes[self.max_idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Max class: {}-{}".format(self.max_cls, list(map(lambda x: round(x, self.decimal), self.scores.tolist()))),
                    (50, 50), font, 2, self.colors[self.max_idx], 3)
        self.pred_cls, self.pred_conf = self.get_pred()
        cv2.putText(img, "Predicted class: {}-{}".format(self.pred_cls, round(self.pred_conf.item(), self.decimal)),
                    (50, 150), font, 2, self.colors[self.max_idx], 3)
        return img

    def release(self):
        return
