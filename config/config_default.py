# -*- coding:utf-8 -*-
from __future__ import print_function
device = "cuda:0"
computer = "laptop"

# Training
metric_names = ["loss", "acc", "auc", "pr"]
metric_directions = ["down", "up", "up", "up"]
cls_metric_names = ["acc", "auc", "pr"]

warm_up = {0: 0.1, 1: 0.5}
bad_epochs = {30: 0.1}
patience_decay = {1: 0.5}

# Testing
label_path = "config/labels/cat_dog.txt"
model_path = "weights/test/log/best_acc.pth"
backbone = "mobilenet"
visualize = True

# Evaluation
eval_model_path = "weight/catdog/mobilenet/default_mobilenet_2cls_best.pth"
eval_img_folder = "data/CatDog"
eval_config = None
eval_keyword = "test"
