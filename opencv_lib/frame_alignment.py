import os
import shutil

input_images_folder = "/media/hkuit164/Backup/xjl/20240111_reverse/selected_frames/OF_selected_frame/normal"
align_images_folder = "/media/hkuit164/Backup/xjl/20240111_reverse/total_frames/raw_total_frame/normal"
output_images_folder = "/media/hkuit164/Backup/xjl/20240111_reverse/selected_frames/raw_selected_frame/normal"

# mode = "irao", "ioar", "ioao"
# irao: input_raw align_other
# ioar: input_other align_raw
# ioao: input_other align_other

mode = "ioar"

input_images_names = [os.path.join(input_images_folder, input_images_name) for input_images_name in os.listdir(input_images_folder)]
align_images_names = [os.path.join(align_images_folder, align_images_name) for align_images_name in os.listdir(align_images_folder)]
os.makedirs(output_images_folder, exist_ok=True)
idx = 0


def data_split(original_name, raw_data):
    img_base_name = os.path.basename(original_name)[:-4]
    name_parts = img_base_name.rsplit("_", 1)
    if raw_data is True:
        base_name = name_parts[0]
    else:
        base_name = (name_parts[0].rsplit("_", 1))[0]
    frame_num = name_parts[1]
    return base_name, frame_num


for input_images_name in input_images_names:

    if mode == "irao":
        input_image_basename, input_frame_num = data_split(input_images_name, raw_data=True)
        for align_images_pathname in align_images_names:
            align_image_basename, align_frame_num = data_split(align_images_pathname, raw_data=False)
            if align_frame_num == input_frame_num and align_image_basename == input_image_basename:
                shutil.copyfile(align_images_pathname, os.path.join(output_images_folder, os.path.basename(align_images_pathname)))
                idx += 1

    elif mode == "ioar":
        input_image_basename, input_frame_num = data_split(input_images_name, raw_data=False)
        for align_images_pathname in align_images_names:
            align_image_basename, align_frame_num = data_split(align_images_pathname, raw_data=True)
            if align_frame_num == input_frame_num and align_image_basename == input_image_basename:
                shutil.copyfile(align_images_pathname, os.path.join(output_images_folder, os.path.basename(align_images_pathname)))
                idx += 1

    else:
        input_image_basename, input_frame_num = data_split(input_images_name, raw_data=False)
        for align_images_pathname in align_images_names:
            align_image_basename, align_frame_num = data_split(align_images_pathname, raw_data=False)
            if align_frame_num == input_frame_num and align_image_basename == input_image_basename:
                shutil.copyfile(align_images_pathname, os.path.join(output_images_folder, os.path.basename(align_images_pathname)))
                idx += 1

print("Input images: {}".format(len(input_images_names)))
print("Aligned images: {}". format(idx))
