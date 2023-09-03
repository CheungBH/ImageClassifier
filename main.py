#-*-coding:utf-8-*-
from models.build import ModelBuilder
from dataset.dataloader import DataLoader
from trainer.optimizer import OptimizerInitializer
from trainer.scheduler import SchedulerInitializer
from trainer.criterion import CriteriaInitializer
from trainer.utils import resume
from eval.evaluate import EpochEvaluator, BatchEvaluator
from logger.record import TrainRecorder

import torch
from tqdm import tqdm
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False

import config.config as config
metric_names = config.metric_names
metric_directions = config.metric_directions
cls_metric_names = config.cls_metric_names


def main(args):
    if args.resume:
        args = resume(args)

    device = args.device

    epochs = args.epochs
    sparse = args.sparse
    backbone = args.backbone

    if backbone != "inception":
        inp_size = 224
        is_inception = False
    else:
        inp_size = 299
        is_inception = True

    iterations = args.iteration

    data_loader = DataLoader()
    data_loader.build_with_args(args, inp_size)
    args.cls_num = data_loader.cls_num
    args.labels = data_loader.label

    MB = ModelBuilder()
    model = MB.build_with_args(args)
    if args.backbone != "vit":
        (args.flops, args.params, args.inf_time) = MB.get_benchmark(inp_size)
    else:
        (args.flops, args.params, args.inf_time) = 0, 0, 0

    criterion = CriteriaInitializer().get(args)
    optimizer = OptimizerInitializer().get(args, MB.params_to_update)
    schedule = SchedulerInitializer()
    schedule.get(args, optimizer)

    TR = TrainRecorder(args, metric_names, metric_directions, cls_metric_names)

    if mix_precision:
        m, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.evaluate:
        validate()


    for epoch in range(epochs)[args.start_epoch:]:
        for phase in ["train", "val"]:
            EpochEval = EpochEvaluator(data_loader.cls_num)
            BatchEval = BatchEvaluator()
            model.train() if phase == "train" else model.eval()

            loader_desc = tqdm(data_loader.dataloaders_dict[phase])

            for i, (names, inputs, labels) in enumerate(loader_desc):

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
                        outputs = MB.softmax(outputs)
                        loss = criterion(outputs, labels)

                if phase == 'train':
                    iterations += 1
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
                    schedule.update(phase, "iter")

                EpochEval.update(outputs, labels, loss)
                batch_loss, batch_acc, batch_auc, batch_pr = BatchEval.update(loss, outputs, labels)
                loader_desc.set_description(
                    '{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                        format(phase=phase, epoch=epoch, loss=batch_loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
                )

            loss, acc, auc, pr, cls_metric = EpochEval.calculate()
            TR.update(model, (loss, acc, auc, pr), epoch, phase, cls_metric)
        schedule.update(phase, "epoch")
        args.iterations = iterations
        args.start_epoch = epoch
        args.train_loss, args.train_acc, args.train_auc, args.train_pr, args.val_loss, args.val_acc, args.val_auc, \
            args.val_pr = TR.get_best_metrics()
        TR.save_option(args)
        print("------------------------------------------------------------------------")
    TR.release()


def train(data_loader, model, optimizer, criterion, is_inception=False, device="cuda:0", iterations=0, sparse=0,
          epoch=-1, schedule=None):
    EpochEval = EpochEvaluator(data_loader.cls_num)
    BatchEval = BatchEvaluator()
    model.train()

    loader_desc = tqdm(data_loader.dataloaders_dict["train"])

    for i, (names, inputs, labels) in enumerate(loader_desc):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if is_inception:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                # outputs = MB.softmax(outputs)
                loss = criterion(outputs, labels)

            iterations += 1
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
            schedule.update("train", "iter")

        EpochEval.update(outputs, labels, loss)
        batch_loss, batch_acc, batch_auc, batch_pr = BatchEval.update(loss, outputs, labels)
        loader_desc.set_description(
            '{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
            format(phase="train", epoch=epoch, loss=batch_loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
        )


def validate(data_loader, model, criterion, is_inception=False, device="cuda:0", iterations=0, sparse=0,
          epoch=-1, schedule=None):
    EpochEval = EpochEvaluator(data_loader.cls_num)
    BatchEval = BatchEvaluator()
    model.eval()

    loader_desc = tqdm(data_loader.dataloaders_dict["val"])

    for i, (names, inputs, labels) in enumerate(loader_desc):

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if is_inception:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                # outputs = MB.softmax(outputs)
                loss = criterion(outputs, labels)

            if sparse > 0:
                for mod in model.modules():
                    if isinstance(mod, torch.nn.BatchNorm2d):
                        mod.weight.grad.data.add_(sparse * torch.sign(mod.weight.data))

            schedule.update("val", "iter")
            iterations += 1

        EpochEval.update(outputs, labels, loss)
        batch_loss, batch_acc, batch_auc, batch_pr = BatchEval.update(loss, outputs, labels)
        loader_desc.set_description(
            '{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
            format(phase="val", epoch=epoch, loss=batch_loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
        )

    loss, acc, auc, pr, cls_metric = EpochEval.calculate()
    TR.update(model, (loss, acc, auc, pr), epoch, phase, cls_metric)


if __name__ == '__main__':
    from config.train_args import args
    main(args)
