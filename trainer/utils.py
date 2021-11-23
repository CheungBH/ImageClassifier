#-*-coding:utf-8-*-
import torch
import os


def get_superior_path(path):
    return "/".join(path.replace("\\", "/").split("/")[:-1])


def get_option_path(m_path, option_name="option.pkl"):
    return os.path.join(get_superior_path(m_path), option_name)


def resume(args):
    print("Before resuming:")
    print(args)
    model_path = args.loadModel
    import os
    option_path = get_option_path(model_path)
    if not os.path.exists(option_path):
        raise FileNotFoundError("The file 'option.pkl' does not exist. Can not resume")
    option = torch.load(option_path)

    args.save_dir = option.save_dir
    args.model_idx = args.save_dir.split("/")[-1]
    args.data_name = option.data_path.split("/")[-1]
    args.trainval_ratio = option.trainval_ratio
    args.labels = option.labels

    args.backbone = option.backbone
    args.freeze = option.freeze

    args.batch_size = option.batch_size
    args.epochs = option.epochs
    args.sparse = option.sparse
    args.load_weight = option.load_weight

    args.optMethod = option.optMethod
    args.LR = option.LR
    args.momentum = option.momentum
    args.weightDecay = option.weightDecay
    args.schedule = option.schedule
    args.schedule_gamma = option.schedule_gamma
    args.crit = option.crit

    args.auto = option.auto
    args.flops = option.flops
    args.params = option.params
    args.inf_time = option.inf_time

    args.iterations = option.iterations
    args.start_epoch = option.start_epoch

    args.train_loss, args.train_acc, args.train_auc, args.train_pr, args.val_loss, args.val_acc, args.val_auc, \
    args.val_pr = option.train_loss, option.train_acc, option.train_auc, option.train_pr, option.val_loss, \
                  option.val_acc, option.val_auc, option.val_pr

    print("After resuming")
    print(option)
    print("-------------------------------------------------------------------------")
    return option
