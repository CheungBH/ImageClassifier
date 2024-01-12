import cv2
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--folder',
                        help='folder name',
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args



def cutVideo2Frame(path, interval):
    video_folders = sorted(os.listdir(path))

    for video_folder in video_folders:
        folder_path = os.path.join(path,video_folder)
        ThermalV = cv2.VideoCapture(folder_path)
        i = 0
        while True:
            ret,Thermal= ThermalV.read()
            if ret & (i % interval==0):
                cv2.imwrite('{}.jpg'.format(folder_path.replace(".mp4", "") + "_" + str(i)), Thermal)
            i = i + 1
            if not ret:
                break

if __name__ == '__main__':
    args = parse_args()
    folderPath = args.folder
    cutVideo2Frame(folderPath, 10)
    
