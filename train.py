#-*-coding:utf-8-*-
from models.build import ModelBuilder
from dataset.dataloader import DataLoader
from trainer.optimizer import OptimizerInitializer
from trainer.scheduler import SchedulerInitializer
from trainer.criterion import CriteriaInitializer
from eval.evaluate import EpochEvaluator, MetricCalculator
from logger.record import TrainRecorder

import torch
from tqdm import tqdm
try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False


def train(args):
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

    data_path = args.data_path
    label_path = args.label_path
    batch_size = args.batch_size
    num_worker = args.num_worker
    iterations = args.iteration

    data_loader = DataLoader(data_path, batch_size=batch_size, num_worker=num_worker, inp_size=inp_size,
                             label_path=label_path)
    args.cls_num = data_loader.cls_num
    # MB = ModelBuilder(backbone, data_loader.cls_num, pretrain=True)
    # model = MB.build()

    MB = ModelBuilder()
    model = MB.build_with_args(args)

    criterion = CriteriaInitializer().get(args)
    optimizer = OptimizerInitializer().get(args, MB.params_to_update)
    scheduler = SchedulerInitializer().get(args, optimizer)

    TR = TrainRecorder(args, ["loss", "acc", "auc", "pr"], ["down", "up", "up", "up"])

    if mix_precision:
        m, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    for epoch in range(epochs):
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
                EpochEval.update(outputs, labels, loss)
                batch_acc, batch_auc, batch_pr = BatchEval.calculate_all(outputs, labels)
                loader_desc.set_description(
                    '{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                        format(phase=phase, epoch=epoch, loss=loss, acc=batch_acc, AUC=batch_auc, PR=batch_pr)
                )

            loss, acc, auc, pr, cls_acc, cls_auc, cls_pr = EpochEval.calculate()
            print('{phase}: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                    format(phase=phase, epoch=epoch, loss=loss, acc=acc, AUC=auc, PR=pr))
            TR.update(model, (loss, acc, auc, pr), epoch, phase)

        print("Finish training epoch {}".format(epoch))


if __name__ == '__main__':
    from config.train_args import args
    train(args)
