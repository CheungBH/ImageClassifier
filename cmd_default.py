cmds = [
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/resnet18_size224.yaml --save_dir exp/reverse_3cls/resnet18_size224',
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/resnet18_size112.yaml --save_dir exp/reverse_3cls/resnet18_size112',
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/shufflenet_size224.yaml --save_dir exp/reverse_3cls/shufflenet_size224',
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/shufflenet_size112.yaml --save_dir exp/reverse_3cls/shufflenet_size112',
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/mobilenet_size224.yaml --save_dir exp/reverse_3cls/mobilenet_size224',
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/mobilenet_size112.yaml --save_dir exp/reverse_3cls/mobilenet_size112',
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/squeezenet_size224.yaml --save_dir exp/reverse_3cls/squeezenet_size224',
    'CUDA_VISIBLE_DEVICES=0 python train.py --data_path data/reverse_3cls --batch_size 32 --epochs 50 --cfg_path config/model_cfg_noAug/squeezenet_size112.yaml --save_dir exp/reverse_3cls/squeezenet_size112',

]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    print("Processing cmd {}".format(idx))
    os.system(cmd)