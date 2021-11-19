#-*-coding:utf-8-*-

from frame_process import FrameProcessor
import os
import cv2
import time

image_ext = ["jpg", "jpeg", "webp", "bmp", "png"]
video_ext = ["mp4", "mov", "avi", "mkv", "MP4"]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 12

save_size = (1280, 720)
show_size = (1280, 720)


class Demo:
    def __init__(self, args):
        self.FP = FrameProcessor()
        self.input = args.input_src
        self.output = args.output_src
        self.show = args.show
        self.save_size = save_size
        self.show_size = show_size
        if os.path.isdir(self.input):
            self.demo_type = "image_folder"
            self.input_imgs = [os.path.join(self.input, file_name) for file_name in os.listdir(self.input)]
            if self.output:
                assert os.path.isdir(self.output), "The output should be a folder when the input is a folder!"
                os.makedirs(self.output, exist_ok=True)
                self.output_imgs = [os.path.join(self.output, file_name) for file_name in os.listdir(self.output)]
        elif isinstance(self.input, int):
            self.demo_type = "video"
            self.cap = cv2.VideoCapture(self.input)
            if self.output:
                out_ext = self.output.split(".")[-1]
                assert out_ext in video_ext, "The output should be a video when the input is webcam!"
                self.out = cv2.VideoWriter(self.output, fourcc, fps, save_size)
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
                    assert out_ext in video_ext, "The output should be a video when the input is a video!"
                    self.out = cv2.VideoWriter(self.output, fourcc, fps, save_size)
            else:
                raise ValueError("Unrecognized src: {}".format(self.input))

    def run(self):
        if self.demo_type == "video":
            idx = 0
            while True:
                ret, frame = self.cap.read()
                if ret:
                    time_begin = time.time()
                    self.FP.process(frame, cnt=idx)
                    print("Processing time is {}".format(round(time.time() - time_begin), 4))
                    if self.show:
                        cv2.imshow("result", cv2.resize(frame, self.show_size))
                        cv2.waitKey(1)
                    if self.output:
                        self.out.write(cv2.resize(frame, self.save_size))
                else:
                    self.FP.release()
                    self.cap.release()
                    if self.output:
                        self.out.release()
                    break
                idx += 1
        elif self.demo_type == "image":
            frame = self.input_img
            self.FP.process(frame)
            if self.show:
                cv2.imshow("result", cv2.resize(frame, self.show_size))
                cv2.waitKey(0)
            if self.output:
                cv2.imwrite(self.output, cv2.resize(frame, self.save_size))
            self.FP.release()
        elif self.demo_type == "image_folder":
            for idx, img_name in enumerate(self.input_imgs):
                frame = cv2.imread(img_name)
                self.FP.process(frame, cnt=idx)
                if self.show:
                    cv2.imshow("result", cv2.resize(frame, self.show_size))
                    cv2.waitKey(1000)
                if self.output:
                    cv2.imwrite(self.output_imgs[idx], cv2.resize(frame, self.save_size))
            self.FP.release()
        else:
            raise ValueError


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_src', help="", required=True)
    parser.add_argument('--output_src', help="")
    parser.add_argument('--show', action="store_true")
    parser.add_argument('--save_ratio', default=1, type=float)
    parser.add_argument('--show_ratio', default=1, type=float)
    args = parser.parse_args()
    demo = Demo(args)
    demo.run()
