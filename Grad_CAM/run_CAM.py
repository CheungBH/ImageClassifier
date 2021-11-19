from __future__ import print_function

import os.path as osp
import os
import torch
import torch.hub
from test_model import Tester
from Grad_CAM.grad_cam import GradCAM
from Grad_CAM.utils import save_gradcam,preprocess


def get_gram(model_path,img_path,target_layers):

    device = torch.device("cuda")
    print ("device",device)
    output_path = os.path.join(img_path.replace("data", "res"), model_path.split('/')[-1][:-4])
    os.makedirs(output_path, exist_ok=True)
    MI =Tester(model_path,0.5)
    model = MI.model.CNN_model
    model.to(device)
    model.eval()
    print("Images:")
    image_list = os.listdir(img_path)
    for path in image_list:
        image_path = os.path.join(img_path,path)
        print("\t#{}".format( image_path))
        image, raw_image = preprocess(image_path)
        image = image.unsqueeze(0).to(device)
        gcam = GradCAM(model=model)
        probs, ids = gcam.forward(image)
        gcam.backward(ids=ids[:, [0]])#0表示预测最高概率类别
        for target_layer in target_layers:
            # Grad-CAM
            regions = gcam.generate(target_layer=target_layer)
            #grad cam
            save_gradcam(
                filename=osp.join(output_path,"gradcam--{}-{}.png".format(path.split('.')[0],target_layer)),
                gcam=regions[0, 0],
                raw_image=raw_image,
            )
if __name__ == "__main__":
    model_path = "/media/hkuit164/WD20EJRX/CNN_classification/weight/underwater_action-2_class_resnet34/7/7_resnet34_2cls_best.pth"
    img_path = '/media/hkuit164/WD20EJRX/CNN_classification/data/test'
    # The four residual layers
    target_layers = ["layer1.1.conv2","layer4.1.conv1","layer2.1.conv1","layer3.1.conv1","layer4.1.bn1"]
    get_gram(model_path,img_path,target_layers)
