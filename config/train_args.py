import argparse

parser = argparse.ArgumentParser()
'''data configuration'''
parser.add_argument('--data_path', required=True)
parser.add_argument('--label_path', default="")
parser.add_argument('--trainval_ratio', default=-1, type=float)
parser.add_argument('--data_percentage', '-dp', type=float, default=1)

'''model configuration'''
parser.add_argument('--cfg_path', default="config/model_cfg/mobilenet_all.yaml", type=str)
parser.add_argument('--freeze', default=0, type=int)

'''train configuration'''
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--num_worker', default=0, type=int)
parser.add_argument('--iteration', default=0, type=int)
parser.add_argument('--sparse', default=0, type=float)
parser.add_argument('--load_weight', default="", type=str)
parser.add_argument('--resume', action="store_true")

'''optimize configuration'''
parser.add_argument('--optMethod', default="adam", type=str)
parser.add_argument('--LR', default=0.001, type=float)
parser.add_argument('--weightDecay', default=0, type=float)
parser.add_argument('--momentum', default=0, type=float)
parser.add_argument('--schedule', default="step", type=str)
parser.add_argument('--schedule_gamma', default="")

'''criteria configuration'''
parser.add_argument('--crit', default="CE", type=str)

'''other configuration'''
parser.add_argument('--save_dir', "-s", default="weights", type=str)
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--save_interval', default=20, type=int)
parser.add_argument('--auto', action="store_true")
parser.add_argument('--evaluate', "-e", action="store_true")

args = parser.parse_args()
