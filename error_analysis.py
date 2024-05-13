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
import os
import config.config as config
from utils.utils import load_config

metric_names = config.error_analysis_metrics


def error_analyse(args):
    device = args.device
    model_path = args.model_path
    data_path = args.data_path
    label_path = args.label_path
    num_worker = args.num_worker
    phase = args.phase
    if args.cfg_path is None:
        args.cfg_path = "/".join(args.model_path.split("/")[:-1]) + "config.yaml"
        assert os.path.exists(args.cfg_path), "The config file does not exist!"
    settings = load_config(args.cfg_path)

    backbone = settings["model"]["backbone"]
    inp_size = settings["model"]["input_size"]
    is_inception = False if backbone != "inception" else True


    data_loader = DataLoader(phases=(phase,))
    data_loader.build(data_path, settings, label_path,1, num_worker, shuffle=False,
                      data_percentage=args.data_percentage)
    args.labels = data_loader.label
    criterion = nn.CrossEntropyLoss()

    MB = ModelBuilder()
    model = MB.build(data_loader.cls_num, backbone, device)
    MB.load_weight(model_path)

    model.eval()

    loader_desc = tqdm(data_loader.dataloaders_dict[phase])
    EAR = ErrorAnalyserRecorder(model_path, metric_names, args.auto, logger_path=args.logger_path)

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
        EAR.update(names[0].split("/")[-1], (loss.tolist(), outputs.max().tolist(), outputs[0][labels].tolist()[0],
                                             int(torch.max(outputs, 1)[1] == labels)))
    EAR.release()
    print("Error analysis has been saved in {}".format(args.logger_path))


class AutoErrorAnalyser:
    def __init__(self, data_path, label_path, phase):
        self.data_path = data_path
        self.label_path = label_path if label_path else "''"
        self.phase = phase

    def run(self, model_path, backbone):
        import os
        cmd = "python error_analysis.py --model_path {} --data_path {} --label_path {} --backbone {} --phase {}".format(
            model_path, self.data_path, self.label_path, backbone, self.phase
        )
        os.system(cmd)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--label_path', default="")
    parser.add_argument('--cfg_path', default=None)
    parser.add_argument('--phase', default="val")
    parser.add_argument('--logger_path', default="")
    parser.add_argument('--data_percentage', '-dp', type=float, default=1)

    parser.add_argument('--num_worker', default=0, type=int)
    parser.add_argument('--auto', action="store_true")
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()

    error_analyse(args)