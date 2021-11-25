from .model_storage import ModelSaver
from .txt_log import txtLogger, BNLogger
from .tb_manager import TensorboardManager
from .log_manager import LoggerManager, TestLoggerManager
from .graph import GraphSaver
from .utils import *
from .metric_manger import MetricManager
from .logger import CustomizedLogger
import os


class TrainRecorder:
    def __init__(self, args, metrics=(), directions=(), cls_metrics=(), phases=("train", "val")):
        self.save_interval = args.save_interval
        os.makedirs(args.save_dir, exist_ok=True)

        self.epochs_ls = []
        self.MS = ModelSaver(args.save_dir)
        self.txt_log = txtLogger(args.save_dir, metrics)
        self.bn_log = BNLogger(args.save_dir)
        self.tb_manager = TensorboardManager(args.save_dir, metrics)
        self.logs = LoggerManager(args, metrics, cls_metrics, phases=phases)
        self.graph = GraphSaver(args.save_dir, metrics)
        self.metrics = MetricManager(metrics, cls_metrics, args.cls_num, directions, phases=phases)

    def update(self, model, metrics, epoch, phase, cls_metrics=()):
        ms_epoch = -1 if epoch % self.save_interval != 0 else epoch

        self.txt_log.update(epoch, phase, metrics)
        self.tb_manager.update(metrics, phase, epoch, model)
        updated_metrics = self.metrics.update(metrics, phase, cls_metrics)

        if phase == "val":
            self.epochs_ls.append(epoch)
            bn_ave = calculate_BN(model)
            self.bn_log.update(epoch, bn_ave)
            self.MS.update(model, ms_epoch, updated_metrics)
            metrics_record, cls_metrics_record = self.metrics.get_current_metrics()
            self.logs.update(epoch, metrics_record, cls_metrics_record)

    def release(self):
        metrics_recorder, best_recorder, metrics = self.metrics.release()
        self.logs.release(best_recorder)
        self.graph.process(self.epochs_ls, metrics_recorder)
        print_final_result(best_recorder, metrics)

    def save_option(self, args):
        torch.save(args, os.path.join(args.save_dir, "option.pkl"))

    def get_best_metrics(self):
        return self.metrics.get_best_metrics()


class TestRecorder:
    def __init__(self, args, metrics, cls_metrics, cls_num):
        self.metrics_name = metrics
        self.metrics = MetricManager(metrics, cls_metrics, cls_num, phases=("test", ))
        self.auto = args.auto
        if self.auto:
            self.logger = TestLoggerManager(args, metrics, cls_metrics, phases=("test", ))

    def process(self, metrics, cls_metrics):
        self.metrics.update(metrics, "test", cls_metrics)
        metrics, cls_metrics = self.metrics.get_current_metrics()
        print_final_result(metrics, self.metrics_name, phases=("test", ))
        if self.auto:
            self.logger.update(metrics, cls_metrics)


class ErrorAnalyserRecorder:
    def __init__(self, model_name, metric_names=(), auto=False):
        self.metric_names = metric_names
        self.name = []
        self.records = [[] for _ in range(len(self.metric_names))]
        self.folder = "/".join(model_name.split("/")[:-2])
        self.model_name = model_name
        self.auto = auto

    def update(self, name, metrics):
        self.name.append(name)
        for idx in range(len(metrics)):
            self.records[idx].append(metrics[idx])

    def release(self):
        for idx in range(len(self.metric_names)):
            logger = CustomizedLogger(self.folder, "model_name," + list2str(self.name)+"\n",
                                      "error_analysis_{}".format(self.metric_names[idx]))
            logger.write([self.model_name] + self.records[idx])
        # if not self.auto:

