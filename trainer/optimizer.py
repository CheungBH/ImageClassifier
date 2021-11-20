#-*-coding:utf-8-*-
import torch.optim as optim


class OptimizerInitializer:
    def __init__(self):
        pass

    def get(self, args, params):
        optMethod = args.optMethod
        LR = args.LR
        weightDecay = args.weightDecay
        momentum = args.momentum

        if optMethod == "adam":
            return optim.Adam(params, lr=LR, weight_decay=weightDecay)
        elif optMethod == 'rmsprop':
            return optim.RMSprop(params, lr=LR, momentum=momentum, weight_decay=weightDecay)
        elif optMethod == 'sgd':
            return optim.SGD(params, lr=LR, momentum=momentum, weight_decay=weightDecay)
        else:
            raise ValueError("This optimizer is not supported now")


