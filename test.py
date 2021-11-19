#-*-coding:utf-8-*-

from dataset.utils import image_normalize, read_labels, get_pretrain
from models.build import ModelBuilder
from dataset.dataloader import DataLoader
from eval.evaluate import EpochEvaluator, MetricCalculator
import torch
from tqdm import tqdm
import torch.nn as nn
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False


def test(args):
    device = "cuda:0"

    model_path = args.model_path
    data_path = args.data_path
    label_path = args.label_path
    phase = args.phase
    model_name = args.model_name if args.model_name else get_pretrain(model_path)

    if model_name != "inception":
        inp_size = 224
        is_inception = False
    else:
        inp_size = 299
        is_inception = True

    batch_size = args.batch_size
    num_worker = args.num_worker

    data_loader = DataLoader(data_path, batch_size=batch_size, num_worker=num_worker, inp_size=inp_size, phases=(phase, ),
                             label_path=label_path)
    MB = ModelBuilder(model_name, data_loader.cls_num)
    model = MB.build()
    MB.load_weight(model_path)
    criterion = nn.CrossEntropyLoss()

    EpochEval = EpochEvaluator(data_loader.cls_num)
    BatchEval = MetricCalculator()
    model.eval()

    loader_desc = tqdm(data_loader.dataloaders_dict[phase])

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
                loss = criterion(outputs, labels)

        EpochEval.update(outputs, labels, loss)
        batch_acc, batch_auc, batch_pr = BatchEval.calculate_all(outputs, labels)
        loader_desc.set_description(
            'Test: loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                format(loss=loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
        )

    loss, acc, auc, pr, cls_acc, cls_auc, cls_pr = EpochEval.calculate()
    print('Test: loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
            format(loss=loss, acc=acc, AUC=auc, PR=pr))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--label_path', default="")
    parser.add_argument('--model_name', default="")
    parser.add_argument('--phase', default="val")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_worker', default=1, type=int)
    args = parser.parse_args()

    test(args)