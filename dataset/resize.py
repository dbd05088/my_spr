import os
import cv2
import numpy as np
path = "./cifar10/train"
klasses = os.listdir(path)

for klass in klasses:
    file_path = path + '/' + klass
    files = os.listdir(file_path)
    for f in files:
        image_path = file_path + '/' + f
        img = cv2.imread(image_path)
        resize_img = cv2.resize(img, (224, 224))
        cv2.imwrite(image_path, resize_img)
