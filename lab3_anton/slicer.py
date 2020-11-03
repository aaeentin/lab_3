import cv2
import numpy as np

cap = cv2.VideoCapture('media/original videos/4.mp4')

i = 87
while (True):
    ret, img = cap.read()
    if (ret == 0): break
    try:
        image_resize=cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"media/sample/{i}.jpg", image_resize)
        i += 1
    except: pass
cap.release()