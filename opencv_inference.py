#-*-coding:utf-8-*-

from eval.inference import ModelInference
import cv2
import numpy as np
from utils.utils import load_config
import json
from opencv_lib.opencv_class import *


image_ext = ["jpg", "jpeg", "webp", "bmp", "png"]
video_ext = ["mp4", "mov", "avi", "mkv", "MP4"]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = 12


class Demo:
    def __init__(self, args):
        settings = load_config(args.cfg_path)
        self.MI = ModelInference(model_path=args.model_path, label_path=args.label_path, backbone=settings["model"]["backbone"],
                 visualize=args.visualize, device=args.device, inp_size=args.inp_size)
        self.input = args.input_src
        self.output = args.output_src
        self.show = True if args.show_ratio else False
        self.show_ratio = args.show_ratio
        self.save_ratio = args.save_ratio

        with open(args.opencv_cfg, 'r') as ft:
            cfg = json.load(ft)
        self.type = cfg["type"]
        if self.type == "bg":
            self.cv_processor = BackgroundExtractor(args.opencv_cfg)
        elif self.type == "optical_flow":
            self.cv_processor = OpticalFlowProcessor(args.opencv_cfg)
        elif self.type == "merge_channel":
            self.cv_processor = MergeChannelProcessor(args.opencv_cfg)
        else:
            raise NotImplementedError

        self.cap = cv2.VideoCapture(self.input)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.output:
            out_ext = self.output.split(".")[-1]
            assert out_ext in video_ext, "The output should be a video when the input is webcam!"
            # self.save_size = (int(self.save_ratio * self.cap.get(3)), int(self.save_ratio * self.cap.get(4)))
            self.save_size = (int(self.save_ratio * self.cap.get(3)), int(self.save_ratio * self.cap.get(4)) * 2)
            self.out = cv2.VideoWriter(self.output, fourcc, fps, self.save_size, True)

    def run(self, view=False):
        frame_idx = 0

        while True:
            ret, image = self.cap.read()
            frame_idx += 1
            if ret is True:
                processed_frame = self.cv_processor(image)
                if view:
                    cv2.imshow("process_img", processed_frame)
                self.out.write(processed_frame)
                c = cv2.waitKey(1)
                if c == 27:
                    break
            else:
                self.MI.release()
                self.cap.release()
                if self.output:
                    self.out.release()
                break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_src', help="", required=True)
    # rtsp://admin:hkuit155@192.168.1.64:554/Streaming/Channels/101/?transportmode=unicast
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--label_path', default="", required=True)
    # parser.add_argument('--backbone', default="mobilenet")
    parser.add_argument('--cfg_path', default="config/model_cfg/mobilenet_all.yaml", type=str)
    parser.add_argument('--opencv_cfg', default="opencv_lib/cfg_test/optical_flow.json", type=str)

    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--output_src', help="")
    parser.add_argument("--inp_size", default=224)

    parser.add_argument('--save_ratio', default=1, type=float)
    parser.add_argument('--show_ratio', default=1, type=float)
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()
    demo = Demo(args)
    demo.run()
