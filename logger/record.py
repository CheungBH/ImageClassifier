from .model_storage import ModelSaver
from .txt_log import txtLogger, BNLogger
from .logger import CustomizedLogger
import torch
import os


class TrainRecorder:
    def __init__(self, args, metrics=(), directions=()):
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.record_args(args)
        self.directions = directions
        self.metrics = metrics
        self.initial_best = [0 for _ in range(len(metrics))]
        for idx, direct in enumerate(self.directions):
            if direct == "down":
                self.initial_best[idx] = -float("inf")
        self.epochs, self.bn_mean_ls = [], []
        assert len(metrics) == len(directions), "The number of metrics and comparision directions is not equal"
        self.metrics_record = {"train": [[] for _ in range(len(metrics))],
                               "val": [[] for _ in range(len(metrics))]}
        self.best_recorder = {"train": self.initial_best, "val": self.initial_best}
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
                in enumerate(zip(metrics, self.metrics, self.directions, self.metrics_record[phase])):
            if self.compare(record, metric, direction):
                updated_metrics.append(m_name)
                self.best_recorder[idx] = metric
        self.MS.update(model, epoch, updated_metrics)

    def calculate_BN(self, model):
        bn_sum, bn_num = 0, 0
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                bn_num += mod.num_features
                bn_sum += torch.sum(abs(mod.weight))
                # self.tb_writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), self.curr_epoch)
        bn_ave = bn_sum / bn_num
        return bn_ave

    @staticmethod
    def compare(before, after, direction):
        if direction == "up":
            return True if after > before else False
        elif direction == "down":
            return True if after < before else False
        else:
            raise ValueError("Please assign the direction correctly")
