import os
import shutil

input_folder = "../data/multiLabel_CatDog/train/images"  # Provide the path to the folder containing images
output_folder = "../data/multiLabel_CatDog/train/labels"  # Provide the path to the output folder where text files will be created

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    class_label = []
    if filename.lower().find("cat") != -1:
        class_label.append("cat")
    elif filename.lower().find("dog") != -1:
        class_label.append("dog")

    output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
    with open(output_file_path, "w") as file:
        for label in class_label:
            file.write(label + "\n")

print("Text files generated successfully.")