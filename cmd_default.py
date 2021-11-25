cmds = [
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod rmsprop --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/1 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod adam --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/2 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod sgd --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/3 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod rmsprop --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/4 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod adam --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/5 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod sgd --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/6 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone shufflenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod rmsprop --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/7 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone shufflenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod adam --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/8 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone shufflenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod sgd --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/9 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone shufflenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod rmsprop --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/10 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone shufflenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod adam --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/11 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone shufflenet --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod sgd --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/12 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet18 --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod rmsprop --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/13 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet18 --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod adam --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/14 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet18 --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod sgd --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/15 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet18 --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod rmsprop --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/16 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet18 --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod adam --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/17 --auto',
    'CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet18 --data_path /home/hkuit155/Desktop/CNN_classification/data/CatDog --trainval_ratio -1 --freeze 0 --sparse 0 --epochs 20 --batch_size 64 --LR 0.001 --schedule step --schedule_gamma 0 --optMethod sgd --momentum 0 --weightDecay 0 --save_interval 5 --save_dir weights/cat_dog/cat_dog/18 --auto',
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    print("Processing cmd {}".format(idx))
    # os.system(cmd)