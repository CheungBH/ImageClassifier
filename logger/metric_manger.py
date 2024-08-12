#-*-coding:utf-8-*-
import copy
from .utils import compare


class MetricManager:
    def __init__(self, metrics, cls_metrics, cls_num, directions=(), phases=("train", "val")):
        self.cls_num = cls_num
        self.metrics = metrics
        self.directions = directions
        self.cls_metrics = cls_metrics
        self.phases = phases
        if "train" in phases:
            assert len(metrics) == len(directions), "The number of metrics and comparision directions is not equal"

            best_template = [0 for _ in range(len(metrics))]
            for idx, direct in enumerate(self.directions):
                if direct == "down":
                    best_template[idx] = float("inf")
            self.best_recorder = {phase: copy.deepcopy(best_template) for phase in phases}

        cls_metric_template = [[[] for _ in range(self.cls_num)] for _ in range(len(self.cls_metrics))]
        self.metrics_record = {phase: [[] for _ in range(len(metrics))] for phase in phases}
        self.cls_metrics_record = {phase: copy.deepcopy(cls_metric_template) for phase in phases}

    def update(self, metrics, phase, cls_metrics):
        self.updated_metrics = []
        if "train" in self.phases:
            # Need to get best metrics
            for idx, (metric, m_name, direction, record) \
                    in enumerate(zip(metrics, self.metrics, self.directions, self.best_recorder[phase])):
                self.metrics_record[phase][idx].append(metric)
                if compare(record, metric, direction):
                    self.updated_metrics.append(m_name)
                    self.best_recorder[phase][idx] = metric
        else:
            for idx, (metric, m_name) \
                    in enumerate(zip(metrics, self.metrics)):
                self.metrics_record[phase][idx].append(metric)

        # for metric_idx in range(len(self.cls_metrics)):
        #     for cls_idx in range(self.cls_num):
        #         self.cls_metrics_record[phase][metric_idx][cls_idx].append(
        #             cls_metrics[metric_idx][cls_idx]
        #         )
        return self.updated_metrics

    def get_best_metrics(self):
        metrics = []
        for phase in self.phases:
            metrics += self.best_recorder[phase]
        return metrics

    def release(self):
        return self.metrics_record, self.best_recorder, self.metrics

    def get_current_metrics(self):
        return self.metrics_record, self.cls_metrics_record