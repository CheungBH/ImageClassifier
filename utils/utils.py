#-*-coding:utf-8-*-
import os


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


def get_backbone(option_file):
    import torch
    return torch.load(option_file).backbone
