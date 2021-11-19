import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision
import os
device = "cuda:0"

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


def image_normalize(img_name, size=224):
    if isinstance(img_name, str):
        image_array = cv2.imread(img_name)
    else:
        image_array = img_name
    image_array = cv2.resize(image_array, (size, size))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
    image_array = image_array.transpose((2, 0, 1))
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor


def read_labels(label_path):
    with open(label_path, "r") as f:
        return [item[:-1] for item in f.readlines()]


def image2tensorboard(img_name, size=224):
    if isinstance(img_name, str):
        image_array = cv2.imread(img_name)
    else:
        image_array = img_name
    image_array = cv2.resize(image_array, (size, size))
    image_array = image_array / 255.0
    image_array = image_array.transpose((2, 0, 1))
    return torch.from_numpy(image_array).float()


def draw_graph(epoch_ls, train_loss_ls, val_loss_ls, train_acc_ls, val_acc_ls, log_dir):
    ln1, = plt.plot(epoch_ls, train_loss_ls, color='red', linewidth=3.0, linestyle='--')
    ln2, = plt.plot(epoch_ls, val_loss_ls, color='blue', linewidth=3.0, linestyle='-.')
    plt.title("Loss")
    plt.legend(handles=[ln1, ln2], labels=['train_loss', 'val_loss'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.savefig(os.path.join(log_dir, "loss.jpg"))
    plt.cla()

    ln1, = plt.plot(epoch_ls, train_acc_ls, color='red', linewidth=3.0, linestyle='--')
    ln2, = plt.plot(epoch_ls, val_acc_ls, color='blue', linewidth=3.0, linestyle='-.')
    plt.title("Acc")
    plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.savefig(os.path.join(log_dir, "acc.jpg"))


def print_model_param_nums(model, multiply_adds=True):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def print_model_param_flops(model=None, input_height=224, input_width=224, multiply_adds=True):
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (
            2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    if device != "cpu":
        input = Variable(torch.rand(3, 3, input_width, input_height).cuda(), requires_grad=True)
    else:
        input = Variable(torch.rand(3, 3, input_width, input_height), requires_grad=True)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(
        list_upsample))
    # print('  + Number of FLOPs: %.5fG' % (total_flops / 3 / 1e9))

    return total_flops / 3


def get_inference_time(model, repeat=190, height=416, width=416):
    model.eval()
    start = time.time()
    with torch.no_grad():
        inp = torch.randn(1, 3, height, width)
        if device != "cpu":
            inp = inp.cuda()
        for i in range(repeat):
            output = model(inp)
    avg_infer_time = (time.time() - start) / repeat

    return round(avg_infer_time, 4)


def generate_cmd(ls):
    string = ""
    for idx, item in enumerate(ls):
        string += item
        string += " "
    return string[:-1] + "\n"


# def adjust_lr(optimizer, epoch, nEpoch):
#     curr_ratio = epoch/nEpoch
#     bound = list(lr_decay.keys())
#     if curr_ratio > bound[0] and curr_ratio <= bound[1]:
#         lr = opt.LR * lr_decay[bound[0]]
#     elif curr_ratio > bound[1]:
#         lr = opt.LR * lr_decay[bound[1]]
#     else:
#         lr = opt.LR
#
#     for pg in optimizer.param_groups:
#         pg["lr"] = lr
#     return optimizer, lr


def write_decay_title(num, char):
    char = char[:-1]
    for n in range(num):
        char += ",decay"
        char += str(n+1)
    char += "\n"
    return char


def write_decay_info(decays, char):
    char = char[:-1]
    for d in decays:
        char += ","
        char += str(d)
    char += "\n"
    return char


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model=None):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def reset(self, new_patience):
        self.counter = 0
        self.patience = new_patience
        self.early_stop = False

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def log_of_each_class(acc):
    tmp = [" "]
    for idx, a in enumerate(acc):
        tmp.append("{}".format(a))
    return tmp


def csv_cls_num(dl):
    tmp = [""] * 6 + ["Total num: "]
    for phase in ["train", "val"]:
        tmp.append(" ")
        ls = []
        loader = dl[phase]
        for k, v in loader.dataset.label_nums.items():
            ls.append(v)
        tmp += ls
    return tmp

def get_pretrain(model_path):
    if "_resnet18" in model_path:
        name = "resnet18"
    elif "_resnet50" in model_path:
        name = "resnet50"
    elif "_resnet34" in model_path:
        name = "resnet34"
    elif "_resnet101" in model_path:
        name = "resnet101"
    elif "_resnet152" in model_path:
        name = "resnet152"
    elif "_inception" in model_path:
        name = "inception"
    elif "_mobilenet" in model_path:
        name = "mobilenet"
    elif "_shufflenet" in model_path:
        name = "shufflenet"
    elif "_LeNet" in model_path:
        name = "LeNet"
    elif "_squeezenet" in model_path:
        name = "squeezenet"
    elif "mnasnet" in model_path:
        name = "mnasnet"
    elif "LeNet" in model_path:
        name = "LeNet"
    else:
        raise ValueError("Wrong name of pre-train model")
    return name