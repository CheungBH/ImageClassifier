#-*-coding:utf-8-*-

import os
from models.build import ModelBuilder
from dataset.dataloader import DataLoader
from eval.evaluate import EpochEvaluator, BatchEvaluator
from logger.record import TestRecorder
import torch
from tqdm import tqdm
import torch.nn as nn
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False
from utils.utils import load_config
import config.config as config
metric_names = config.metric_names
cls_metric_names = config.cls_metric_names


def test(args):
    device = args.device

    model_path = args.model_path
    data_path = args.data_path
    if args.cfg_path is None:
        args.cfg_path = os.path.join(os.path.dirname(args.model_path), "config.yaml")
        assert os.path.exists(args.cfg_path), "The config file does not exist!"
    settings = load_config(args.cfg_path)

    batch_size = args.batch_size
    num_worker = args.num_worker
    phase = args.phase
    backbone = settings["model"]["backbone"]
    inp_size = settings["model"]["input_size"]
    is_inception = False if backbone != "inception" else True


    data_loader = DataLoader(phases=(phase,))
    data_loader.build(data_path, settings, "", batch_size, num_worker)
    args.labels = data_loader.label

    MB = ModelBuilder()
    model = MB.build(data_loader.cls_num, backbone, args.device)
    MB.load_weight(model_path)
    criterion = nn.CrossEntropyLoss()

    EpochEval = EpochEvaluator(data_loader.cls_num)
    BatchEval = BatchEvaluator()
    model.eval()

    loader_desc = tqdm(data_loader.dataloaders_dict[phase])
    TR = TestRecorder(args, metric_names, cls_metric_names, data_loader.cls_num)

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

        EpochEval.update(outputs, labels, loss)
        batch_loss, batch_acc, batch_auc, batch_pr = BatchEval.update(loss, outputs, labels)
        loader_desc.set_description(
            'Test: loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                format(loss=batch_loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
        )

    loss, acc, auc, pr, cls_metrics = EpochEval.calculate()
    EpochEval.generate_confusion_matrix(plot=True, labels=args.labels)
    TR.process([loss, acc, auc, pr], cls_metrics)


class AutoTester:
    def __init__(self, data_path, label_path, phase):
        self.data_path = data_path
        self.label_path = label_path if label_path else "''"
        self.phase = phase

    def run(self, model_path, backbone):
        import os
        cmd = "python test.py --model_path {} --data_path {} --label_path {} --backbone {} --phase {} --auto".format(
            model_path, self.data_path, self.label_path, backbone, self.phase
        )
        os.system(cmd)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--cfg_path', default=None)
    parser.add_argument('--phase', default="val")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--auto', action="store_true")
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()

    test(args)