from .model_storage import ModelSaver
from .txt_log import txtLogger, BNLogger
from .logger import CustomizedLogger
import torch
import os
import copy


class TrainRecorder:
    def __init__(self, args, metrics=(), directions=(), cls_metrics=()):
        self.save_dir = args.save_dir
        self.cls_num = args.cls_num
        self.cls_metrics = cls_metrics
        os.makedirs(self.save_dir, exist_ok=True)
        self.record_args(args)
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
        self.epochs, self.bn_mean_ls = [], []
        self.MS = ModelSaver(self.save_dir)
        self.txt_log = txtLogger(self.save_dir, self.metrics)
        self.bn_log = BNLogger(self.save_dir)

    def record_args(self, args):
        self.model_idx = self.save_dir.split("/")[-1]
        self.epochs = args.epochs
        self.sparse = args.sparse
        self.save_interval = args.save_interval
        self.data_path = args.data_path
        self.label_path = args.label_path
        self.batch_size = args.batch_size
        self.num_worker = args.num_worker
        self.iterations = args.iteration

    def update(self, model, metrics, epoch, phase, cls_metrics=()):
        self.epochs.append(epoch)
        bn_ave = self.calculate_BN(model)
        self.bn_mean_ls.append(bn_ave)
        self.txt_log.update(epoch, phase, metrics)
        self.bn_log.update(epoch, bn_ave)

        epoch = -1 if epoch % self.save_interval != 0 else epoch
        updated_metrics = []
        for idx, (metric, m_name, direction, record) \
                in enumerate(zip(metrics, self.metrics, self.directions, self.best_recorder[phase])):
            self.metrics_record[phase][idx].append(metric)
            if self.compare(record, metric, direction):
                updated_metrics.append(m_name)
                self.best_recorder[idx] = metric
        self.MS.update(model, epoch, updated_metrics)

        for metric_idx in range(len(self.cls_metrics)):
            for cls_idx in range(self.cls_num):
                self.cls_metrics_record[phase][metric_idx][cls_idx].append(
                    cls_metrics[metric_idx][cls_idx]
                )

    def calculate_BN(self, model):
        bn_sum, bn_num = 0, 0
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                bn_num += mod.num_features
                bn_sum += torch.sum(abs(mod.weight))
                # self.tb_writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), self.curr_epoch)
        bn_ave = bn_sum / bn_num
        return bn_ave.tolist()

    @staticmethod
    def compare(before, after, direction):
        if direction == "up":
            return True if after > before else False
        elif direction == "down":
            return True if after < before else False
        else:
            raise ValueError("Please assign the direction correctly")
