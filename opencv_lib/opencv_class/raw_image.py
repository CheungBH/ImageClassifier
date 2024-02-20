import cv2 as cv


class RawImageProcessor:
    def __init__(self, cfg_file):
        pass

    def __call__(self, frame, view=False):
        if view:
            cv.imshow("raw", frame)
        return frame
