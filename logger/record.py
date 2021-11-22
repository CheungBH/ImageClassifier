from .model_storage import ModelSaver
from .txt_log import txtLogger, BNLogger
from .tb_manager import TensorboardManager
from .log_manager import LoggerManager
from .utils import *
import os
import copy


class TrainRecorder:
    def __init__(self, args, metrics=(), directions=(), cls_metrics=()):
        self.save_dir = args.save_dir
        self.save_interval = args.save_interval

        self.cls_num = args.cls_num
        self.cls_metrics = cls_metrics
        os.makedirs(self.save_dir, exist_ok=True)
        self.directions = directions
        self.metrics = metrics
        assert len(metrics) == len(directions), "The number of metrics and comparision directions is not equal"
        best_template = [0 for _ in range(len(metrics))]
        for idx, direct in enumerate(self.directions):
            if direct == "down":
                best_template[idx] = float("inf")

        self.metrics_record = {"train": [[] for _ in range(len(metrics))],
                               "val": [[] for _ in range(len(metrics))]}
        self.best_recorder = {"train": best_template, "val": best_template}
        cls_metric_template = [[[] for _ in range(self.cls_num)] for _ in range(len(self.cls_metrics))]
        self.cls_metrics_record = {"train": copy.deepcopy(cls_metric_template),
                                   "val": copy.deepcopy(cls_metric_template)}
        self.epochs_ls, self.bn_mean_ls = [], []
        self.MS = ModelSaver(self.save_dir)
        self.txt_log = txtLogger(self.save_dir, self.metrics)
        self.bn_log = BNLogger(self.save_dir)
        self.tb_manager = TensorboardManager(self.save_dir, self.metrics)
        self.logs = LoggerManager(args, self.metrics, self.cls_metrics)

    def update(self, model, metrics, epoch, phase, cls_metrics=()):
        self.epochs_ls.append(epoch)
        self.txt_log.update(epoch, phase, metrics)
        self.tb_manager.update(metrics, phase, epoch, model)

        epoch = -1 if epoch % self.save_interval != 0 else epoch
        updated_metrics = []
        for idx, (metric, m_name, direction, record) \
                in enumerate(zip(metrics, self.metrics, self.directions, self.best_recorder[phase])):
            self.metrics_record[phase][idx].append(metric)
            if compare(record, metric, direction):
                updated_metrics.append(m_name)
                self.best_recorder[idx] = metric
        self.MS.update(model, epoch, updated_metrics)

        for metric_idx in range(len(self.cls_metrics)):
            for cls_idx in range(self.cls_num):
                self.cls_metrics_record[phase][metric_idx][cls_idx].append(
                    cls_metrics[metric_idx][cls_idx]
                )

        if phase == "train":
            bn_ave = calculate_BN(model)
            self.bn_mean_ls.append(bn_ave)
            self.bn_log.update(epoch, bn_ave)
        elif phase == "val":
            self.logs.update(self.epochs_ls[-1], self.metrics_record, self.cls_metrics_record)
