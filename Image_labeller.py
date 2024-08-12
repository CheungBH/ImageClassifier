import os
import cv2
import tkinter as tk
from tkinter import messagebox
import json
from PIL import Image, ImageTk

folder_path = "videos/catdog_sample"  # Specify the folder path directly
output_path = "videos/catdog_sample_labeled.json"  # Specify the output JSON file path directly

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.image_files = []
        self.current_index = 0
        self.classification_results = {}
        self.create_widgets()

        self.load_labels()
        self.load_images()

    def load_labels(self):
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                self.classification_results = json.load(f)

    def create_widgets(self):
        self.root.title("Image Classifier")

        self.buttons_frame_top = tk.Frame(self.root)
        self.buttons_frame_top.pack(side=tk.TOP, pady=10)

        self.back_button = tk.Button(self.buttons_frame_top, text="Back", command=self.previous_image)
        self.back_button.pack(side=tk.LEFT, padx=5)
        self.back_button.config(state=tk.DISABLED)

        self.next_button = tk.Button(self.buttons_frame_top, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=5)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)

        self.label_frame = tk.Frame(self.root)
        self.label_frame.pack(pady=10)

        self.label_input = tk.Entry(self.label_frame)
        self.label_input.pack(side=tk.LEFT, padx=5)

        self.set_label_button = tk.Button(self.label_frame, text="Set Label", command=self.label_image)
        self.set_label_button.pack(side=tk.LEFT)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        self.current_label_display = tk.Label(self.root, text="Current Label: None")
        self.current_label_display.pack(pady=10)

    def load_images(self):
        self.image_files = [
            file for file in os.listdir(folder_path)
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(self.image_files) == 0:
            messagebox.showerror("Error", "No image files found in the selected folder.")
            return

        self.display_image()

    def display_image(self):
        image_path = os.path.join(folder_path, self.image_files[self.current_index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        max_height = 500
        max_width = 600
        image_height, image_width, _ = image.shape
        if image_height > max_height or image_width > max_width:
            scale = min(max_height / image_height, max_width / image_width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        self.photo = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(self.photo)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

        current_image = self.image_files[self.current_index]
        if current_image.split('.')[0] in self.classification_results:
            self.label_input.delete(0, tk.END)
            self.label_input.insert(0, self.classification_results[current_image.split('.')[0]])
            self.current_label_display.config(text=f"Current Label: {self.classification_results[current_image.split('.')[0]]:.2f}")
        else:
            self.label_input.delete(0, tk.END)
            self.current_label_display.config(text="Current Label: None")

    def label_image(self):
        try:
            label_value = float(self.label_input.get())
            if 0 <= label_value <= 1:
                current_image = self.image_files[self.current_index]
                self.classification_results[current_image.split('.')[0]] = label_value
                self.current_label_display.config(text=f"Current Label: {label_value:.2f}")
            else:
                messagebox.showerror("Error", "Label must be between 0 and 1.")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter a number between 0 and 1.")

    def next_image(self):
        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.display_image()
            self.back_button.config(state=tk.NORMAL)
        else:
            self.save_results()
            messagebox.showinfo("Info", "Image labeling complete.")
            self.root.destroy()

    def previous_image(self):
        self.current_index -= 1
        if self.current_index >= 0:
            self.display_image()
            if self.current_index == 0:
                self.back_button.config(state=tk.DISABLED)

    def save_results(self):
        with open(output_path, "w") as f:
            json.dump(self.classification_results, f, indent=4)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageClassifierGUI(root)
    root.mainloop()