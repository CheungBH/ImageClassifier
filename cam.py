import argparse
import os
import cv2
import numpy as np
import torch
from utils.utils import load_config
from models.build import ModelBuilder
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default="data/3cls/val/Finish",
        help='Input image folder path')
    parser.add_argument(
        '--model_path',
        type=str,
        default="weights/3cls/280.pth",
        help='Path to model')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component' 
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='eigengradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output/eigengradcam',
                        help='Output directory to save the images')
    args = parser.parse_args()

    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args

def process_single_image(image_path, model, cam, args, gb_model):
    """处理单张图片的函数"""
    # 读取和预处理图像
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]).to(args.device)

    # 获取预测结果
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # 生成CAM
    grayscale_cam = cam(input_tensor=input_tensor,
                       targets=None,
                       aug_smooth=args.aug_smooth,
                       eigen_smooth=args.eigen_smooth)
    grayscale_cam = grayscale_cam[0, :]

    # 生成可视化
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # Guided backprop
    gb = gb_model(input_tensor, target_category=None)
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    return cam_image, gb, cam_gb, predicted_class

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        # "scorecam": ScoreCAM,
        # "gradcam++": GradCAMPlusPlus,
        # "ablationcam": AblationCAM,
        # "xgradcam": XGradCAM,
        # "eigencam": EigenCAM,
        # "eigengradcam": EigenGradCAM,
        # "layercam": LayerCAM,
        # "fullgrad": FullGrad,
        # "gradcamelementwise": GradCAMElementWise,
        # 'kpcacam': KPCA_CAM
    }

    model_path = args.model_path
    cfg_path = "/".join(args.model_path.split("/")[:-1]) + "/config.yaml"
    assert os.path.exists(cfg_path), "The config file does not exist!"
    settings = load_config(cfg_path)
    backbone = settings["model"]["backbone"]
    MB = ModelBuilder()
    model = MB.build(3, backbone, args.device)
    MB.load_weight(model_path)

    target_layers = [model.features[-1]]

    cams = {}
    for method_name, method in methods.items():
        cams[method_name] = [method(model=model, target_layers=target_layers),
                             GuidedBackpropReLUModel(model=model, device=args.device)]

    if os.path.isdir(args.image_path):
        image_files = [f for f in os.listdir(args.image_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        image_files = [os.path.basename(args.image_path)]
        args.image_path = os.path.dirname(args.image_path)


    for img_file in image_files:
        sub_folder_path = os.path.join(args.output_dir, img_file.split(".")[0])
        os.makedirs(sub_folder_path, exist_ok=True)
        full_path = os.path.join(args.image_path, img_file)
        print(f"Processing {img_file}...")

        for cam_name, (cam, gb_model) in cams.items():
            # try:
                cam_image, gb, cam_gb, predicted_class = process_single_image(
                    full_path, model, cam, args, gb_model)
                # torch.cuda.empty_cache()
                base_name = os.path.splitext(img_file)[0]
                cv2.imwrite(os.path.join(sub_folder_path, f'{base_name}_{cam_name}.jpg'), cam_image)
                print(f"Image: {img_file}, Predicted class: {predicted_class}")
            # except:
            #     print("Error when forwarding {}".format(cam_name))

