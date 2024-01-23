#-*-coding:utf-8-*-

from eval.inference import ModelInference
import cv2
import time
from utils.utils import load_config

image_ext = ["jpg", "jpeg", "webp", "bmp", "png"]
video_ext = ["mp4", "mov", "avi", "mkv", "MP4"]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 12


class Demo:
    def __init__(self, args):
        settings = load_config(args.cfg_path)
        self.MI = ModelInference(model_path=args.model_path, label_path=args.label_path, backbone=settings["model"]["backbone"],
                 visualize=args.visualize, device=args.device)
        self.input = args.input_src
        self.output = args.output_src
        self.show = True if args.show_ratio else False
        self.show_ratio = args.show_ratio
        self.save_ratio = args.save_ratio
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        self.cap = cv2.VideoCapture(self.input)
        if self.output:
            out_ext = self.output.split(".")[-1]
            assert out_ext in video_ext, "The output should be a video when the input is webcam!"
            self.save_size = (int(self.save_ratio * self.cap.get(3)), int(self.save_ratio * self.cap.get(4)))
            self.out = cv2.VideoWriter(self.output, fourcc, fps, self.save_size)

    def run(self):
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                time_begin = time.time()
                fgmask = self.mog.apply(frame)
                ret, binary = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.se)
                input_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                # bgimage = self.mog.getBackgroundImage()
                self.MI.run(input_img, cnt=idx)
                print("Processing time is {}".format(round(time.time() - time_begin), 4))
                if self.show:
                    show_size = (int(self.show_ratio * frame.shape[1]), int(self.show_ratio * frame.shape[0]))
                    cv2.imshow("result", cv2.resize(frame, show_size))
                    cv2.imshow("input", cv2.resize(input_img, show_size))
                    cv2.waitKey(1)
                if self.output:
                    self.out.write(cv2.resize(frame, self.save_size))
            else:
                self.MI.release()
                self.cap.release()
                if self.output:
                    self.out.release()
                break
            idx += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_src', help="", required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--label_path', default="", required=True)
    # parser.add_argument('--backbone', default="mobilenet")
    parser.add_argument('--cfg_path', default="config/model_cfg/mobilenet_all.yaml", type=str)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--output_src', help="")

    parser.add_argument('--save_ratio', default=1, type=float)
    parser.add_argument('--show_ratio', default=1, type=float)
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()
    demo = Demo(args)
    demo.run()
