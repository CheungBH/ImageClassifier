#-*-coding:utf-8-*-

from eval.inference import ModelInference
import os
import cv2
import time
from utils.utils import load_config

image_ext = ["jpg", "jpeg", "webp", "bmp", "png"]
video_ext = ["mp4", "mov", "avi", "mkv", "MP4"]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 12


class Demo:
    def __init__(self, args):
        if args.cfg_path is None:
            args.cfg_path = "/".join(args.model_path.split("/")[:-1]) + "/config.yaml"
            assert os.path.exists(args.cfg_path), "The config file does not exist!"
        settings = load_config(args.cfg_path)
        backbone = settings["model"]["backbone"]
        inp_size = settings["model"]["input_size"]

        self.MI = ModelInference(model_path=args.model_path, backbone=backbone, visualize=args.visualize,
                                 device=args.device, inp_size=inp_size, conf=args.conf)
        self.input = args.input_src
        self.output = args.output_src
        self.show = True if args.show_ratio else False
        self.show_ratio = args.show_ratio
        self.save_ratio = args.save_ratio

        if os.path.isdir(self.input):
            self.demo_type = "image_folder"
            self.input_imgs = [os.path.join(self.input, file_name) for file_name in os.listdir(self.input)]
            if self.output:
                os.makedirs(self.output, exist_ok=True)
                assert os.path.isdir(self.output), "The output should be a folder when the input is a folder!"
                os.makedirs(self.output, exist_ok=True)
                self.output_imgs = [os.path.join(self.output, file_name) for file_name in os.listdir(self.input)]
        elif isinstance(self.input, int):
            self.demo_type = "video"
            self.cap = cv2.VideoCapture(self.input)
            if self.output:
                out_ext = self.output.split(".")[-1]
                assert out_ext in video_ext, "The output should be a video when the input is webcam!"
                self.save_size = (int(self.save_ratio * self.cap.get(3)), int(self.save_ratio * self.cap.get(4)))
                self.out = cv2.VideoWriter(self.output, fourcc, fps, self.save_size)
        else:
            ext = self.input.split(".")[-1]
            if ext in image_ext:
                self.demo_type = "image"
                self.input_img = cv2.imread(self.input)
                if self.output:
                    out_ext = self.output.split(".")[-1]
                    assert out_ext in image_ext, "The output should be an image when the input is an image!"
            elif ext in video_ext:
                self.demo_type = "video"
                self.cap = cv2.VideoCapture(self.input)
                if self.output:
                    out_ext = self.output.split(".")[-1]
                    self.save_size = (int(self.save_ratio * self.cap.get(3)), int(self.save_ratio * self.cap.get(4)))
                    assert out_ext in video_ext, "The output should be a video when the input is a video!"
                    self.out = cv2.VideoWriter(self.output, fourcc, fps, self.save_size)
            else:
                raise ValueError("Unrecognized src: {}".format(self.input))

    def run(self):
        if self.demo_type == "video":
            idx = 0
            while True:
                ret, frame = self.cap.read()
                if ret:
                    time_begin = time.time()
                    self.MI.run(frame, cnt=idx)
                    print("Processing time is {}".format(round(time.time() - time_begin), 4))
                    if self.show:
                        show_size = (int(self.show_ratio * frame.shape[1]), int(self.show_ratio * frame.shape[0]))
                        cv2.imshow("result", cv2.resize(frame, show_size))
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
        elif self.demo_type == "image":
            frame = self.input_img
            self.MI.run(frame)
            if self.show:
                show_size = (int(self.show_ratio * frame.shape[1]), int(self.show_ratio *frame.shape[0]))
                cv2.imshow("result", cv2.resize(frame, show_size))
                cv2.waitKey(0)
            if self.output:
                save_size = (int(self.save_ratio * frame.shape[1]), int(self.save_ratio *frame.shape[0]))
                cv2.imwrite(self.output, cv2.resize(frame, save_size))
            self.MI.release()
        elif self.demo_type == "image_folder":
            for idx, img_name in enumerate(self.input_imgs):
                frame = cv2.imread(img_name)
                self.MI.run(frame, cnt=idx)
                if self.show:
                    show_size = (int(self.show_ratio * frame.shape[1]), int(self.show_ratio * frame.shape[0]))
                    cv2.imshow("result", cv2.resize(frame, show_size))
                    cv2.waitKey(0)
                if self.output:
                    save_size = (int(self.save_ratio * frame.shape[1]), int(self.save_ratio * frame.shape[0]))
                    cv2.imwrite(self.output_imgs[idx], cv2.resize(frame, save_size))
            self.MI.release()
        else:
            raise ValueError


class AutoDemo:
    def __init__(self, input_src, out_src, label_path):
        self.input_src = input_src
        self.label_path = label_path
        self.out_src = os.path.join(out_src, "demo")

    def run(self, model_path, backbone):
        import os
        out_folder = os.path.join(self.out_src, model_path.split("/")[-2])
        os.makedirs(out_folder, exist_ok=True)
        cmd = "python demo.py --model_path {} --input_src {} --output_src {} --label_path {} --backbone {} " \
              "--show_ratio 0 --visualize".format(
            model_path, self.input_src, out_folder, self.label_path, backbone
        )
        os.system(cmd)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_src', help="Target input", required=True)
    parser.add_argument('--model_path', required=True)
    # parser.add_argument('--label_path', )
    parser.add_argument('--cfg_path', default=None)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--output_src', help="")
    parser.add_argument('--conf', default=0.5, type=float)

    parser.add_argument('--save_ratio', default=1, type=float)
    parser.add_argument('--show_ratio', default=1, type=float)
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()
    demo = Demo(args)
    demo.run()
