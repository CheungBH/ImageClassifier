import os
import random
import shutil


def random_select_images(source_folder, target_folder, num_images):
    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Randomly select 'num_images' number of files
    selected_files = random.sample(files, num_images)

    # Copy selected files to the target folder
    for file in selected_files:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        shutil.copyfile(source_path, target_path)
        print(f"Copied {file} to {target_folder}")


# Specify the paths and number of images to select
source_folder = "../data/CatDog/val/dog"
target_folder = "../data/multiLabel_CatDog/val/images"
os.makedirs(target_folder, exist_ok=True)

num_images = 100
# val_images = 100
random_select_images(source_folder, target_folder, num_images)