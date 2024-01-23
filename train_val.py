import os
import random
import shutil


def rename_and_sort_images(source_folder, ratio, folder1_path, folder2_path):

    image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    random.shuffle(image_files)

    split_index = int(len(image_files) * ratio)

    os.makedirs(folder1_path, exist_ok=True)

    os.makedirs(folder2_path, exist_ok=True)

    for i, image_file in enumerate(image_files):

        file_name = os.path.basename(image_file)
        new_filename = f"{file_name}"

        source_path = os.path.join(source_folder, image_file)
        
        if i < split_index:
            destination_path = os.path.join(folder1_path, new_filename)
        else:
            destination_path = os.path.join(folder2_path, new_filename)

        shutil.copyfile(source_path, destination_path)


source_folder = '/media/hkuit164/Backup/xjl/reverse_cls/r/reverse_of'
split_ratio = 0.8
folder1_path = 'data/normal_reverse_of_new/train/reverse'
folder2_path = 'data/normal_reverse_of_new/val/reverse'

rename_and_sort_images(source_folder, split_ratio, folder1_path, folder2_path)
