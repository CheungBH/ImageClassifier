import cv2
import numpy as np
import onnxruntime as rt


def image_process(image_path):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # (96, 96, 3)

    image = img.astype(np.float32) / 255.0
    image = (image - mean) / std

    image = image.transpose((2, 0, 1))  # (3, 96, 96)
    image = image[np.newaxis, :, :, :]  # (1, 3, 96, 96)

    image = np.array(image, dtype=np.float32)

    return image


def onnx_runtime(img_path, model_path):
    imgdata = image_process(img_path)

    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    pred_onnx = sess.run([output_name], {input_name: imgdata})

    print("outputs:")
    print(np.array(pred_onnx))


onnx_runtime('../data/CatDog/val/cat/cat.31.jpg', '../model_sim.onnx')
