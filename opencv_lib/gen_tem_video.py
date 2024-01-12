import cv2 as cv
import json
import numpy as np
import argparse
import os
import shutil

bg_cfg = "cfg/bg_extract.json"
OF_cfg_ft = "cfg/optical_flow_feature.json"
OF_cfg_lk = "cfg/optical_flow_lk.json"


def bg_extract(video_input, video_output, view):
    cap = cv.VideoCapture(video_input)
    output_file = video_output

    with open(bg_cfg, 'r') as config_file:
        config = json.load(config_file)

    mog = cv.createBackgroundSubtractorMOG2()
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # se = cv.getStructuringElement(**config)

    output_fps = cap.get(cv.CAP_PROP_FPS)
    output_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file, fourcc, output_fps, output_size, isColor=False)

    while True:
        ret, image = cap.read()
        if ret is True:
            fgmask = mog.apply(image)
            ret, binary = cv.threshold(fgmask, 220, 255, cv.THRESH_BINARY)
            binary = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
            bgimage = mog.getBackgroundImage()
            if view:
                cv.imshow("bgimage", bgimage)
                # cv.imshow("frame", image)
                cv.imshow("fgmask", binary)

            out.write(binary)

            c = cv.waitKey(50)
            if c == 27:
                break
        else:
            break

    out.release()
    cv.destroyAllWindows()


def optical_flow(video_input, video_output, view):
    cap = cv.VideoCapture(video_input)
    output_path = video_output

    with open(OF_cfg_ft, 'r') as ft:
        config_ft = json.load(ft)
    with open(OF_cfg_lk, 'r') as lk:
        config_lk = json.load(lk)


    #角点检测参数
    # feature_params = dict(maxCorners =100, qualityLevel=0.1, minDistance=7, blockSize=7)
    feature_params = dict(**config_ft)
    #KLT光流参数
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))
    # lk_params = dict(**config_lk)

    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv.CAP_PROP_FPS)

    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (np.int(width), np.int(height)), True)

    tracks = []
    track_len = 15
    frame_idx = 0
    detect_interval = 5
    while True:

        ret, frame = cap.read()
        if ret:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis_black = np.zeros_like(frame)
            vis = frame.copy()

            if len(tracks)>0:
                img0 ,img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
                # 上一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                # 反向检查,当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                # 得到角点回溯与前一帧实际角点的位置变化关系
                d = abs(p0-p0r).reshape(-1,2).max(-1)

                #判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                good = d < 1

                new_tracks = []

                for i, (tr, (x, y), flag) in enumerate(zip(tracks, p1.reshape(-1, 2), good)):

                    # 判断是否为正确的跟踪点
                    if not flag:
                        continue

                    # 存储动态的角点
                    tr.append((x, y))

                    # 只保留track_len长度的数据，消除掉前面的超出的轨迹
                    if len(tr) > track_len:
                        del tr[0]
                    # 保存在新的list中
                    new_tracks.append(tr)

                    cv.circle(vis_black, (int(x), int(y)), 3, (255, 0, 0), 3, 1)

                # 更新特征点
                tracks = new_tracks

                # #以上一振角点为初始点，当前帧跟踪到的点为终点,画出运动轨迹
                cv.polylines(vis_black, [np.int32(tr) for tr in tracks], False, (0, 255, 0), 3)

            # 每隔 detect_interval 时间检测一次特征点
            if frame_idx % detect_interval==0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255

                if frame_idx !=0:
                    for x,y in [np.int32(tr[-1]) for tr in tracks]:
                        cv.circle(mask, (x, y), 5, 0, -1)

                p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1,2):
                        tracks.append([(x, y)])

            frame_idx += 1
            prev_gray = frame_gray
            if view:
                cv.imshow('track', frame)
                cv.imshow("raw", vis_black)
            out.write(vis_black)
            #out.write(vis)
            ch = cv.waitKey(1)
            if ch ==27:
                cv.imwrite('track.jpg', vis)
                break
        else:
            break

    cv.destroyAllWindows()
    cap.release()


def video_merge(video_input, buffer, video_output):
    out = video_output
    bg_extract(video_input, f"{buffer}/1.mp4", view=False)
    optical_flow(video_input, f"{buffer}/2.mp4", view=False)

    # 3 to 1
    bg = cv.VideoCapture(f"{buffer}/1.mp4")
    of = cv.VideoCapture(f"{buffer}/2.mp4")
    rgb = cv.VideoCapture(video_input)
    frame_width = int(rgb.get(3))
    frame_height = int(rgb.get(4))
    fps = rgb.get(cv.CAP_PROP_FPS)
    merge_video = cv.VideoWriter(out, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret1, frame1 = bg.read()
        ret2, frame2 = of.read()
        ret3, frame3 = rgb.read()
        if ret1 and ret2 and ret3:
            gray_1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            gray_2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            gray_3 = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)

            merge_frame = cv.merge((gray_1, gray_2, gray_3))
            merge_video.write(merge_frame)
        else:
            break

    bg.release()
    of.release()
    rgb.release()
    merge_video.release()
    cv.destroyAllWindows()
    shutil.rmtree(buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_video_folder", type=str, help="Path to raw video folder")
    parser.add_argument("--output_video_folder", type=str, help="Path to generated video folder")
    parser.add_argument("--bg_extract", action='store_true', help="Generate bg_extract videos")
    parser.add_argument("--optical_flow", action='store_true', help="Generate optical_flow videos")
    parser.add_argument("--view_videos", action='store_true', help="View videos while generating")

    parser.add_argument("--video_channel_merge", action='store_true', help="Combine 3 different video channels")

    opt = parser.parse_args()
    input_video_paths = [os.path.join(opt.raw_video_folder, input_video_path) for input_video_path in os.listdir(opt.raw_video_folder)]
    for input_video_path in input_video_paths:
        video_name = os.path.basename(input_video_path)[:-4]

        if opt.bg_extract:
            output_video_path = f"{opt.output_video_folder}/{video_name}_bg.mp4"
            bg_extract(input_video_path,  output_video_path, opt.view_videos)
        if opt.optical_flow:
            output_video_path = f"{opt.output_video_folder}/{video_name}_OF.mp4"
            optical_flow(input_video_path, output_video_path, opt.view_videos)
        if opt.video_channel_merge:
            os.makedirs(os.path.join(opt.output_video_folder, "buffer"), exist_ok=True)
            output_video_path = f"{opt.output_video_folder}/{video_name}_combined.mp4"
            video_merge(input_video_path, f"{opt.output_video_folder}/buffer", output_video_path)


if __name__ == "__main__":
    main()
