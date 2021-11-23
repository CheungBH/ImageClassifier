#-*-coding:utf-8-*-
from sklearn import metrics
import torch


class CurveLogger:
    def __init__(self):
        self.clear()

    def clear(self):
        self.gt = []
        self.preds = []

    def update(self, gt, preds):
        if len(self.gt) == 0:
            self.gt = gt
            self.preds = preds
        else:
            self.gt = torch.cat((self.gt, gt))
            self.preds = torch.cat((self.preds, preds))

    def cal_AUC(self):
        try:
            auc = metrics.roc_auc_score(self.preds, self.gt)
        except:
            auc = 0
        return auc


class DataLogger:
    def __init__(self):
        self.clear()

    def clear(self):
        self.sum = 0
        self.cnt = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.cnt += n

    def average(self):
        return self.sum / self.cnt


class MetricCalculator:
    def __init__(self):
        pass

    def cal_acc(self, outputs, labels):
        return ((torch.max(outputs, 1)[1] == labels).sum()).tolist() /len(outputs)

    def cal_auc(self, preds, labels):
        try:
            auc = metrics.roc_auc_score(preds, labels)
        except:
            auc = 0
        return auc

    def cal_PR(self, preds, labels):
        try:
            P, R, thresh = metrics.precision_recall_curve(preds, labels)
            area = 0
            for idx in range(len(thresh)-1):
                a = (R[idx] - R[idx+1]) * (P[idx+1] + P[idx])/2
                area += a
            return area
        except:
            return 0

    def get_thresh(self, preds, labels):
        try:
            P, R, thresh = metrics.precision_recall_curve(preds, labels)
            PR_ls = [P[idx] + R[idx] for idx in range(len(P))]
            max_idx = PR_ls.index(max(PR_ls))
            return thresh[max_idx]
        except:
            return 0

    def calculate_all(self, outputs, labels):
        acc = self.cal_acc(outputs, labels)
        _, preds = torch.max(outputs, 1)
        preds, labels = preds.detach().cpu(), labels.detach().cpu()
        auc = self.cal_auc(preds, labels)
        pr = self.cal_PR(preds, labels)
        return acc, auc, pr


