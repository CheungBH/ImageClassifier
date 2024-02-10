
import cv2 as cv
import json
from .utils import str2tuple


class BackgroundExtractor:
    def __init__(self, cfg_file):
        with open(cfg_file, 'r') as ft:
            cfg = json.load(ft)
        self.mog = cv.createBackgroundSubtractorMOG2()
        self.se = cv.getStructuringElement(cv.MORPH_RECT, str2tuple(cfg["ksize"]))
        self.thresh = cfg["thresh"]

    def __call__(self, frame, view=False):
        fgmask = self.mog.apply(frame)
        ret, binary = cv.threshold(fgmask, self.thresh, 255, cv.THRESH_BINARY)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, self.se)
        bgimage = self.mog.getBackgroundImage()
        if view:
            cv.imshow("fgmask", bgimage)
        return cv.cvtColor(binary, cv.COLOR_GRAY2BGR)