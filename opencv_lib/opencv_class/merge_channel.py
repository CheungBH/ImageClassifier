from .bg_extract import BackgroundExtractor
from .raw_image import RawImageProcessor
from .optical_flow import OpticalFlowProcessor
import json
import os
import cv2 as cv

processor_dict = {
    "raw": RawImageProcessor,
    "bg": BackgroundExtractor,
    "optical": OpticalFlowProcessor
}


class MergeChannelProcessor:

    def __init__(self, cfg_file):
        base_path = "/".join(cfg_file.split("/")[:-1])
        with open(cfg_file, "r") as cfg_file:
            cfgs = json.load(cfg_file)
        sequences, files = cfgs["sequence"], cfgs["files"]
        self.processors = []

        for sequence, file in zip(sequences, files):
            assert sequence in ["raw", "bg", "optical"]
            self.processors.append(processor_dict[sequence](os.path.join(base_path, file)))

    def __call__(self, frame, view=False):
        gray_imgs = []
        for processor in self.processors:
            gray_imgs.append(cv.cvtColor(processor(frame, view=view), cv.COLOR_BGR2GRAY))
        return cv.merge(tuple(gray_imgs))
