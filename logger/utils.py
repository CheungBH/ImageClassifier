import torch


def calculate_BN(model):
    bn_sum, bn_num = 0, 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            bn_num += mod.num_features
            bn_sum += torch.sum(abs(mod.weight))
            # self.tb_writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), self.curr_epoch)
    bn_ave = bn_sum / bn_num
    return bn_ave.tolist()


def compare(before, after, direction):
    if direction == "up":
        return True if after > before else False
    elif direction == "down":
        return True if after < before else False
    else:
        raise ValueError("Please assign the direction correctly")
