#-*-coding:utf-8-*-
import torch
from sklearn import metrics


class BatchEvaluator:
    def __init__(self):
        pass


class EpochEvaluator:
    def __init__(self):
        self.preds, self.labels = [], []
        self.MC = MetricCalculator()
        self.loss = 0

    def update(self, preds, labels, loss):
        self.loss += loss
        if len(self.preds) == 0:
            self.preds = preds
            self.labels = labels
        else:
            self.preds = torch.cat((self.preds, preds), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def calculate(self):
        loss = self.loss / len(self.labels)
        acc = self.MC.cal_acc(self.preds, self.labels)
        auc = self.MC.cal_auc(self.preds, self.labels)
        pr = self.MC.cal_PR(self.preds, self.labels)
        return loss, acc, auc, pr


class MetricCalculator:
    def __init__(self):
        pass

    def cal_acc(self, preds, labels):
        return ((torch.max(preds, 1)[1] == labels).sum()).tolist() /len(preds)

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



