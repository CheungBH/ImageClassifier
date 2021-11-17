#-*-coding:utf-8-*-

from models.build import ModelBuilder
from dataset.dataloader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import torch
import torch.nn as nn
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False

device = "cuda:0"

epochs = 20

optMethod = "adam"
LR = 0.001
weightDecay = 0
momentum = 0
schedule = "step"

model_name = "mobilenet"
freeze = 0

if model_name != "inception":
    inp_size = 224
    is_inception = False
else:
    inp_size = 299
    is_inception = True

data_path = "/home/hkuit155/Desktop/CNN_classification/data/CatDog"
batch_size = 64
num_worker = 2

data_loader = DataLoader(data_path, batch_size=batch_size, num_worker=num_worker, inp_size=inp_size)
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


for epoch in range(epochs):
    for phase in ["train", "val"]:
        model.train() if phase == "train" else model.eval()
        loss_sum = torch.zeros(1)

        for names, inputs, labels in data_loader.dataloaders_dict[phase]:

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

                _, preds = torch.max(outputs, 1)

            print("Current loss is {}".format(loss))

            if phase == 'train':
                if mix_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
