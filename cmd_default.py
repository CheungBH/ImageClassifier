cmds = [
    'python train_model.py --backbone shufflenet --batch 32 --loadModel pre_train_model/shufflenet.pth --epoch 50 --expFolder catdog --expID shuffule --dataset CatDog',
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    print("Processing cmd {}".format(idx))
    os.system(cmd)
