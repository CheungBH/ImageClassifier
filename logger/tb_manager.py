
from tensorboardX import SummaryWriter


class TensorboardManager:
    def __init__(self, folder):
        self.tb_writer = SummaryWriter(folder)




