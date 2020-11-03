import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('classified2.avi',fourcc, 20.0, (768,1024))

for i in range(213):
	img = cv2.imread(f"media/nikolay/frame_cl/frame{i}.jpg")
	out.write(img)
out.release()