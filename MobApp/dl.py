import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import easyocr


model = tf.keras.models.load_model('./static/models/resnetV2.h5')


def detect(path, filename):
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(224, 224))
    image_arr = img_to_array(image1) / 255.0
    h, w, d = image.shape
    test_arr = image_arr.reshape(1, 224, 224, 3)

    coords = model.predict(test_arr)

    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)

    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)

    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 3)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
    return image, coords

def OCR(path, filename):
    image, cd = detect(path, filename)
    xmin, xmax, ymin, ymax = cd[0][0], cd[0][1], cd[0][2], cd[0][3]
    cropped_img = image[ymin:ymax, xmin:xmax]
    cropped_BGR = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename), cropped_BGR)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(cropped_img)
    if results:
        detected_text = results[0][1]
    return detected_text