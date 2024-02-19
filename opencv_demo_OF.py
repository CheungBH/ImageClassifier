#-*-coding:utf-8-*-

from eval.inference import ModelInference
import cv2
import numpy as np
from utils.utils import load_config

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
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.track_len = 15
        self.detect_interval = 5
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

        self.cap = cv2.VideoCapture(self.input)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.output:
            out_ext = self.output.split(".")[-1]
            assert out_ext in video_ext, "The output should be a video when the input is webcam!"
            # self.save_size = (int(self.save_ratio * self.cap.get(3)), int(self.save_ratio * self.cap.get(4)))
            self.save_size = (int(self.save_ratio * self.cap.get(3)), int(self.save_ratio * self.cap.get(4)) * 2)
            self.out = cv2.VideoWriter(self.output, fourcc, fps, self.save_size, True)

    def run(self):
        idx = 0
        tracks = []
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis_black = np.zeros_like(frame)
                vis = frame.copy()
                if len(tracks)>0:
                    img0 ,img1 = prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                    p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                    d = abs(p0-p0r).reshape(-1,2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for i, (tr, (x, y), flag) in enumerate(zip(tracks, p1.reshape(-1, 2), good)):
                        if not flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis_black, (int(x), int(y)), 3, (255, 0, 0), 3, 1)
                    tracks = new_tracks
                    cv2.polylines(vis_black, [np.int32(tr) for tr in tracks], False, (0, 255, 0), 3)

                if idx % self.detect_interval==0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255

                    if idx !=0:
                        for x,y in [np.int32(tr[-1]) for tr in tracks]:
                            cv2.circle(mask, (x, y), 5, 0, -1)

                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1,2):
                            tracks.append([(x, y)])
                self.MI.run(vis_black, cnt=idx)
                idx += 1
                prev_gray = frame_gray
                cv2.imshow('track', vis)
                cv2.imshow("raw", vis_black)
                cv2.waitKey(1)
                canvas = np.concatenate((vis, vis_black), axis=0)
                if args.output_src:
                    self.out.write(canvas)

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
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--output_src', help="")
    parser.add_argument("--inp_size", default=224)

    parser.add_argument('--save_ratio', default=1, type=float)
    parser.add_argument('--show_ratio', default=1, type=float)
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()
    demo = Demo(args)
    demo.run()
