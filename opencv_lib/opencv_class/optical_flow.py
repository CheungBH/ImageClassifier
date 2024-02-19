import json
import cv2 as cv
from .utils import str2tuple
import numpy as np


class OpticalFlowProcessor:
    def __init__(self, cfg_file):
        with open(cfg_file, 'r') as ft:
            cfg = json.load(ft)

        # self.maxCorners = cfg["maxCorners"]
        # self.qualityLevel = cfg["qualityLevel"]
        # self.minDistance = cfg["minDistance"]
        # self.blockSize = cfg["blockSize"]
        # self.maxLevel = cfg["maxLevel"]
        # self.winSize = str2tuple(cfg["winSize"])
        # self.criteria = cfg["criteria"]
        self.feature_params = dict(maxCorners=cfg["maxCorners"], qualityLevel=cfg["qualityLevel"],
                                   minDistance=cfg["minDistance"], blockSize=cfg["blockSize"])
        self.lk_params = dict(winSize=str2tuple(cfg["winSize"]), maxLevel=cfg["maxLevel"],
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))
        self.tracks = []
        self.track_len = 15
        self.frame_idx = 0
        self.detect_interval = 5

    def __call__(self, frame, view=False):

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vis_black = np.zeros_like(frame)
        # tracks = []

        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)

            good = d < 1

            new_tracks = []

            for i, (tr, (x, y), flag) in enumerate(zip(self.tracks, p1.reshape(-1, 2), good)):

                if not flag:
                    continue

                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)

                cv.circle(vis_black, (int(x), int(y)), 3, (255, 0, 0), 3, 1)

            self.tracks = new_tracks
            cv.polylines(vis_black, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0), 3)

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            if self.frame_idx != 0:
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)

            p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        self.frame_idx += 1
        self.prev_gray = frame_gray
        if view:
            cv.imshow('track', frame)
        return vis_black

