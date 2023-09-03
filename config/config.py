# -*- coding:utf-8 -*-
from __future__ import print_function
computer = "laptop"

# Training
metric_names = ["loss", "acc", "auc", "pr"]
metric_directions = ["down", "up", "up", "up"]
cls_metric_names = ["acc", "auc", "pr"]

warm_up = {0: 0.1, 1: 0.5}
bad_epochs = {30: 0.1}
patience_decay = {1: 0.5}


error_analysis_metric = ["loss", "accuracy"]
