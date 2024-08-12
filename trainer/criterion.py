import torch.nn as nn


class CriteriaInitializer:
    def __init__(self):
        pass

    def get(self, args):
        crit = args.crit
        if crit == "CE":
            return nn.CrossEntropyLoss()
        elif crit == "ML":
            return nn.MultiLabelSoftMarginLoss()
        elif crit == "BCE":
            return nn.BCEWithLogitsLoss()
        elif crit == "MSE":
            return nn.MSELoss()
        else:
            raise NotImplementedError("Current criteria '{}' is not supported yet".format(crit))


