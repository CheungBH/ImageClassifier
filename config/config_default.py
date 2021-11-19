# -*- coding:utf-8 -*-
from __future__ import print_function
from src.opt import opt
device = "cuda:0"
computer = "laptop"

# Training
datasets = {"ceiling": ["freestyle", "frog", "side_freestyle", "side_frog", "butterfly", "back", "standing"],
            "CatDog": ["cat", "dog"],
            "squat": ["standing", "squat"],
            "squat_cut": ["standing", "squating"],
            "drown_stand": ["drown", "stand_walk"],
            "exercise": ["push_up", "sit_up", "squat_up"]}

if opt.backbone == 'inception':
    input_size = 299
else:
    input_size = 224

freeze_pretrain = {"mobilenet": [155, "classifier"],
                   "shufflenet": [167, "fc"],
                   "mnasnet": [155, "classifier"],
                   "resnet18": [59, "fc"],
                   "squeezenet": [49, "classifier"],
                   "resnet34": [107, "fc"],
                   }

warm_up = {0: 0.1, 1: 0.5}
bad_epochs = {30: 0.1}
patience_decay = {1: 0.5}

# Testing
test_model_path = "weight/test/finetune/finetune_resnet18_2cls_best.pth"
test_img_folder = "data/CatDog"
test_config = "prune/cfg2.txt"

# Evaluation
eval_model_path = "weight/test/origin/origin_resnet18_2cls_best.pth"
eval_img_folder = "data/CatDog"
eval_config = None
eval_keyword = "val"
