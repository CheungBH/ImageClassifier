#-*-coding:utf-8-*-
from models.build import ModelBuilder
from dataset.dataloader import DataLoader
from trainer.optimizer import OptimizerInitializer
from trainer.scheduler import SchedulerInitializer
from trainer.criterion import CriteriaInitializer
from trainer.utils import resume
from eval.evaluate import EpochEvaluator, MetricCalculator
from logger.record import TrainRecorder
import os

import torch
from tqdm import tqdm
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False


def train(args):
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
    (args.flops, args.params, args.inf_time) = MB.get_benchmark(inp_size)

    criterion = CriteriaInitializer().get(args)
    optimizer = OptimizerInitializer().get(args, MB.params_to_update)
    schedule = SchedulerInitializer()
    schedule.get(args, optimizer)

    TR = TrainRecorder(args, ["loss", "acc", "auc", "pr"], ["down", "up", "up", "up"], ["acc", "auc", "pr"])

    if mix_precision:
        m, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    for epoch in range(epochs)[args.start_epoch:]:
        for phase in ["train", "val"]:
            EpochEval = EpochEvaluator(data_loader.cls_num)
            BatchEval = MetricCalculator()
            model.train() if phase == "train" else model.eval()

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
                        outputs = MB.softmax(outputs)
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
                    schedule.update(phase, "iter")

                EpochEval.update(outputs, labels, loss)
                batch_acc, batch_auc, batch_pr = BatchEval.calculate_all(outputs, labels)
                loader_desc.set_description(
                    '{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                        format(phase=phase, epoch=epoch, loss=loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
                )

            schedule.update(phase, "epoch")
            loss, acc, auc, pr, cls_metric = EpochEval.calculate()
            TR.update(model, (loss, acc, auc, pr), epoch, phase, cls_metric)
        args.iterations = iterations
        args.start_epoch = epoch
        args.train_loss, args.train_acc, args.train_auc, args.train_pr, args.val_loss, args.val_acc, args.val_auc, \
            args.val_pr = TR.get_best_metrics()
        torch.save(args, os.path.join(args.save_dir))
        print("------------------------------------------------------------------------")
    TR.release()


if __name__ == '__main__':
    from config.train_args import args
    train(args)
