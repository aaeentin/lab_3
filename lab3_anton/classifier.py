import tensorflow as tf
K = tf.keras
import numpy as np
#from imim import *
import cv2

model = K.models.load_model("my_model")


def classify_img(img):
	image_resize=cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
	image_resize.resize((1, image_resize.shape[0], image_resize.shape[1], image_resize.shape[2]))
	y_pred_test = model.predict_proba(image_resize)
	return np.argmax(y_pred_test, axis=1)



fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('partner_Sample2.avi',fourcc, 20.0, (1080,1920))

cap = cv2.VideoCapture('nikolay_sample/norezinka.mp4')


ret, img = cap.read()
cl = classify_img(img)

while (True):
	ret, img = cap.read()
	if (ret == 0): break
	try:
		cl = classify_img(img)
		if cl == 1:
			img = cv2.putText(img, 'pivo', (540, 960), cv2.FONT_HERSHEY_SIMPLEX,\
				1, (255,0,0), 2, cv2.LINE_AA)
			out.write(img)
		else:
			img = cv2.putText(img, 'ne pivo', (540, 960), cv2.FONT_HERSHEY_SIMPLEX,\
				1, (255,0,0), 2, cv2.LINE_AA)
			out.write(img)
	except: 
		out.write(img)
cap.release()
out.release()