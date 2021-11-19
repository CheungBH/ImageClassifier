#-*-coding:utf-8-*-
import os
from models.build import ModelBuilder
from dataset.dataloader import DataLoader
from eval.evaluate import EpochEvaluator, MetricCalculator
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import torch
from tqdm import tqdm
import torch.nn as nn
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False

device = "cuda:0"
save_dir = "weights/test/1"

epochs = 20

optMethod = "adam"
LR = 0.001
weightDecay = 0
momentum = 0
schedule = "step"
sparse = 0

model_name = "mobilenet"
freeze = 0

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
iterations = 0

data_loader = DataLoader(data_path, batch_size=batch_size, num_worker=num_worker, inp_size=inp_size,
                         label_path=label_path)
MB = ModelBuilder(model_name, data_loader.cls_num, pretrain=True)
model = MB.build()
criterion = nn.CrossEntropyLoss()

if optMethod == "adam":
    optimizer = optim.Adam(MB.params_to_update, lr=LR, weight_decay=weightDecay)
elif optMethod == 'rmsprop':
    optimizer = optim.RMSprop(MB.params_to_update, lr=LR, momentum=momentum, weight_decay=weightDecay)
elif optMethod == 'sgd':
    optimizer = optim.SGD(MB.params_to_update, lr=LR, momentum=momentum, weight_decay=weightDecay)
else:
    raise ValueError("This optimizer is not supported now")

if schedule == "step":
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs*0.7), int(epochs*0.9)], gamma=0.1)
elif schedule == "exp":
    scheduler = ExponentialLR(optimizer, gamma=0.9999)
elif schedule == "stable":
    scheduler = None
else:
    raise NotImplementedError("The scheduler is not supported")


if mix_precision:
    m, optimizer = amp.initialize(model, optimizer, opt_level="O1")

os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    for phase in ["train", "val"]:
        EpochEval = EpochEvaluator(data_loader.cls_num)
        BatchEval = MetricCalculator()
        model.train() if phase == "train" else model.eval()
        loss_sum = torch.zeros(1)

        loader_desc = tqdm(data_loader.dataloaders_dict[phase])

        for i, (names, inputs, labels) in enumerate(loader_desc):

            iterations += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                if is_inception and phase == 'train':
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            if phase == 'train':
                if mix_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if sparse > 0:
                    for mod in model.modules():
                        if isinstance(mod, torch.nn.BatchNorm2d):
                            mod.weight.grad.data.add_(sparse * torch.sign(mod.weight.data))

                optimizer.step()
            EpochEval.update(outputs, labels, loss)
            acc, auc, pr = BatchEval.calculate_all(outputs, labels)
            loader_desc.set_description(
                '{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                    format(phase=phase, epoch=epoch, loss=loss, acc=acc, AUC=auc, PR=pr)
            )

        loss, acc, auc, pr, cls_acc, cls_auc, cls_pr = EpochEval.calculate()
        print('{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                format(phase=phase, epoch=epoch, loss=loss, acc=acc, AUC=auc, PR=pr))

    print("Finish training epoch {}".format(epoch))
    torch.save(model.state_dict(), os.path.join(save_dir, "{}.pth".format(epoch)))
