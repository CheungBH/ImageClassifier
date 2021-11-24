#-*-coding:utf-8-*-
import torch
from .logger import MetricCalculator, DataLogger, CurveLogger


class EpochEvaluator:
    def __init__(self, num_cls):
        self.cls_MCs = [MetricCalculator() for _ in range(num_cls)]
        self.outputs, self.labels = [], []
        self.MC = MetricCalculator()
        self.loss = 0

    def update(self, outputs, labels, loss):
        self.loss += loss * len(outputs)
        if len(self.outputs) == 0:
            self.outputs = outputs
            self.labels = labels
        else:
            self.outputs = torch.cat((self.outputs, outputs), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def calculate(self):
        loss = self.loss / len(self.labels)
        acc, auc, pr = self.MC.calculate_all(self.outputs, self.labels)
        cls_acc, cls_auc, cls_pr = self.calculate_cls()
        return loss.tolist(), acc, auc, pr, [cls_acc, cls_auc, cls_pr]

    def calculate_cls(self):
        cls_acc, cls_auc, cls_pr = [], [], []
        for i in range(len(self.cls_MCs)):
            sampled_idx = torch.nonzero(self.labels == i).view(-1).tolist()
            sampled_labels, sampled_outputs = self.labels[sampled_idx], self.outputs[sampled_idx]
            cls_acc.append(self.cls_MCs[i].cal_acc(sampled_outputs, sampled_labels))
            _, sampled_preds = torch.max(sampled_outputs, 1)
            sampled_preds, sampled_labels = sampled_preds.detach().cpu(), sampled_labels.detach().cpu()
            cls_auc.append(self.cls_MCs[i].cal_auc(sampled_outputs, sampled_labels))
            cls_pr.append(self.cls_MCs[i].cal_PR(sampled_outputs, sampled_labels))
        return cls_acc, cls_auc, cls_pr


class BatchEvaluator:
    def __init__(self):
        self.loss_logger = DataLogger()
        self.metric_logger = MetricCalculator()
        self.outputs, self.labels = [], []

    def update(self, loss, outputs, labels):
        self.loss_logger.update(loss, len(outputs))
        if len(self.outputs) == 0:
            self.outputs = outputs
            self.labels = labels
        else:
            self.outputs = torch.cat((self.outputs, outputs), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)
        loss = self.loss_logger.average()
        acc = self.metric_logger.cal_acc(self.outputs, self.labels)
        _, auc, pr = self.metric_logger.calculate_all(self.outputs, self.labels)
        return loss, acc, auc, pr
