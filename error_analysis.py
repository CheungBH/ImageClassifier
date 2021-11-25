#-*-coding:utf-8-*-
#-*-coding:utf-8-*-

from dataset.utils import get_pretrain
from models.build import ModelBuilder
from dataset.dataloader import DataLoader
from logger.record import ErrorAnalyserRecorder
import torch
from tqdm import tqdm
import torch.nn as nn
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False

import config.config as config
metric_names = config.error_analysis_metrics


def error_analyse(args):
    device = "cuda:0"

    model_path = args.model_path
    data_path = args.data_path
    label_path = args.label_path
    num_worker = args.num_worker
    phase = args.phase
    backbone = args.backbone if args.backbone else get_pretrain(model_path)

    if backbone != "inception":
        inp_size = 224
        is_inception = False
    else:
        inp_size = 299
        is_inception = True

    data_loader = DataLoader(phases=(phase,))
    data_loader.build(data_path, label_path, inp_size, 1, num_worker, shuffle=False)
    args.labels = data_loader.label
    criterion = nn.CrossEntropyLoss()

    MB = ModelBuilder()
    model = MB.build(data_loader.cls_num, backbone)
    MB.load_weight(model_path)

    model.eval()

    loader_desc = tqdm(data_loader.dataloaders_dict[phase])
    EAR = ErrorAnalyserRecorder(args, metric_names)

    for i, (names, inputs, labels) in enumerate(loader_desc):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(phase == 'train'):
            if is_inception:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                outputs = MB.softmax(outputs)
                loss = criterion(outputs, labels)

        loader_desc.set_description("Error analysing")
        EAR.update(names[0], (loss, int(torch.max(outputs, 1) == labels)))
    EAR.release()


class AutoErrorAnalyser:
    def __init__(self, data_path, label_path, phase):
        self.data_path = data_path
        self.label_path = label_path if label_path else "''"
        self.phase = phase

    def run(self, model_path, backbone):
        import os
        cmd = "python error_analysis.py --model_path {} --data_path {} --label_path {} --backbone {} --phase {} --auto".format(
            model_path, self.data_path, self.label_path, backbone, self.phase
        )
        os.system(cmd)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--label_path', default="")
    parser.add_argument('--backbone', default="")
    parser.add_argument('--phase', default="val")
    parser.add_argument('--num_worker', default=1, type=int)
    args = parser.parse_args()

    error_analyse(args)