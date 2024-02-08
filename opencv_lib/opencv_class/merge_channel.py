from .bg_extract import BackgroundExtractor
from .raw_image import RawImageProcessor
from .optical_flow import OpticalFlowProcessor
import json
import os
import cv2 as cv

processor_dict = {
    "raw": RawImageProcessor,
    "bg": BackgroundExtractor,
    "optical_flow": OpticalFlowProcessor
}


class MergeChannelProcessor:

    def __init__(self, cfg_file):
        base_path = "/".join(cfg_file.split("/")[:-1])
        cfgs = json.load(cfg_file)
        sequences, files = cfgs["sequence"], cfgs["files"]
        self.processors = []

        for sequence, file in zip(sequences, files):
            assert sequence in ["raw", "bg", "optical_flow"]
            self.processors.append(processor_dict[sequence](os.path.join(file, base_path)))

    def __call__(self, frame):
        gray_imgs = []
        for processor in self.processors:
            gray_imgs.append(cv.cvtColor(processor(frame), cv.COLOR_BGR2GRAY))
        return cv.merge(tuple(gray_imgs))
