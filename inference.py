#-*-coding:utf-8-*-
from dataset.utils import image_normalize, read_labels, get_pretrain
from models.build import ModelBuilder
import cv2
import config.config as config

label_path = config.label_path
model_path = config.model_path
model_name = config.model_name
visualize = config.visualize


class ModelInference:
    def __init__(self):
        self.model_name = model_name if model_name else get_pretrain(model_path)
        self.model_size = 224
        self.classes = read_labels(label_path)
        self.MB = ModelBuilder(self.model_name, len(self.classes))
        self.model = self.MB.build()
        self.MB.load_weight(model_path)
        self.model.eval()
        self.visualize = visualize

    def run(self, img, cnt=0):
        img_tns = image_normalize(img, size=224)
        self.scores = self.MB.inference(img_tns)
        self.pred_idx = self.scores[0].tolist().index(max(self.scores[0].tolist()))
        self.pred_cls = self.classes[self.pred_idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.visualize:
            cv2.putText(img, self.pred_cls, (50, 50), font, 2, (0, 0, 255), 3)
        return img

    def release(self):
        return
