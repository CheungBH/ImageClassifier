# -*- coding:utf-8 -*-
from __future__ import print_function
computer = "laptop"

# Training
metric_names = ["loss", "acc"]
metric_directions = ["down", "up"]
cls_metric_names = ["acc"]

warm_up = {0: 0.1, 1: 0.5}
bad_epochs = {30: 0.1}
patience_decay = {1: 0.5}


error_analysis_metrics = ["loss", "pred_conf", "label_conf", "correct"]
