#-*-coding:utf-8-*-
import os
import yaml


def load_config(config_file):
    with open(config_file, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config


def init_model_list_with_kw(folder, kws, fkws=""):
    valid_folders, models, options = [], [], []

    if not fkws:
        valid_folders = [os.path.join(folder, sub_folder) for sub_folder in os.listdir(folder)
                        if os.path.isdir(os.path.join(folder, sub_folder))]
    else:
        for sub_folder in os.listdir(folder):
            sub_folder_path = os.path.join(folder, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue
            for fkw in fkws:
                if fkw in sub_folder:
                    valid_folders.append(sub_folder_path)

    for valid_folder in valid_folders:
        model_cnt = 0
        for file_name in os.listdir(valid_folder):
            file_path = os.path.join(valid_folder, file_name)
            if "option" in file_name:
                op = file_path
            elif ".pth" in file_name:
                for kw in kws:
                    if kw in file_name:
                        models.append(file_path)
                        model_cnt += 1
            else:
                continue

        for _ in range(model_cnt):
            options.append(op)

    return models, options


def init_model_list(folder):
    models, options = [], []
    for sub_folder in os.listdir(folder):
        sub_folder_path = os.path.join(folder, sub_folder)
        model_cnt = 0
        if "csv" in sub_folder_path:
            continue

        for file_name in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, file_name)
            if "option" in file_name:
                options.append(file_path)
            elif ".pth" in file_path or "pkl" in file_name:
                models.append(file_path)
                model_cnt += 1
                if model_cnt > 1:
                    raise AssertionError("More than one model exist in the folder: {}".format(sub_folder_path))
            else:
                continue
    return models, options


def get_runtime_params(process_type, option_file):
    import torch
    opt = torch.load(option_file)
    if process_type == "test" or process_type == "demo" or process_type == "error_analyse":
        return opt.backbone
    elif process_type == "convert":
        backbone = opt.backbone
        num_cls = opt.cls_num
        inp_size = 299 if backbone == "inception" else 224
        return num_cls, inp_size, backbone


def convert_csv(path):
    import pandas as pd

    df = pd.read_csv(path)
    data = df.values  # data是数组，直接从文件读出来的数据格式是数组
    index1 = list(df.keys())  # 获取原有csv文件的标题，并形成列表
    data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
    data = pd.DataFrame(data, index=index1)  # 将data的行列转换
    data.to_csv(path, header=0)
