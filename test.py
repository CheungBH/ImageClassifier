#-*-coding:utf-8-*-

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

device = "cuda:0"

model_path = "weights/test/1/19.pth"
model_name = "mobilenet"

if model_name != "inception":
    inp_size = 224
    is_inception = False
else:
    inp_size = 299
    is_inception = True

data_path = "/home/hkuit155/Desktop/CNN_classification/data/CatDog"
label_path = ""
batch_size = 64
num_worker = 2

data_loader = DataLoader(data_path, batch_size=batch_size, num_worker=num_worker, inp_size=inp_size, phases=("val", ),
                         label_path=label_path)
MB = ModelBuilder(model_name, data_loader.cls_num)
model = MB.build()
MB.load_weight(model_path)
criterion = nn.CrossEntropyLoss()

phase = "test"
EpochEval = EpochEvaluator(data_loader.cls_num)
BatchEval = MetricCalculator()
model.eval()
loss_sum = torch.zeros(1)

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
        '{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
            format(phase=phase, epoch=0, loss=loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
    )

loss, acc, auc, pr, cls_acc, cls_auc, cls_pr = EpochEval.calculate()
print('{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
        format(phase=phase, epoch=0, loss=loss, acc=acc, AUC=auc, PR=pr))
