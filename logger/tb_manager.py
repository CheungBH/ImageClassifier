import torch
from tensorboardX import SummaryWriter


class TensorboardManager:
    def __init__(self, folder, metrics):
        self.tb_writer = SummaryWriter(folder)
        self.metrics = metrics

    def update(self, metrics, phase, epoch, model):
        for metric, metric_name in zip(metrics, self.metrics):
            self.tb_writer.add_scalar('{}/{}'.format(phase, metric_name), metric, epoch)
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                self.tb_writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), epoch)



