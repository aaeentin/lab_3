import cv2
import numpy as np 

size_pos = 300
size_neg = 150
range_pos_test = range(300, 387)
range_neg_test = range (150, 179)

array_pos = []
for i in range(size_pos):
	img = np.array(cv2.imread(f'media/pos/{i}.jpg'))
	array_pos += [img]
	pass
array_pos = np.array(array_pos)

array_neg = []
for i in range(size_neg):
	img = np.array(cv2.imread(f'media/neg/{i}.jpg'))
	array_neg += [img]
	pass
array_neg = np.array(array_neg)

x_train = np.concatenate((array_pos,array_neg), axis = 0)
y_train = np.ones((x_train.shape[0], 2))
y_train[:array_pos.shape[0], 1] = 0.
y_train[array_pos.shape[0]:, 0] = 0.

x_train2 = x_train/255 - 0.5




array_pos = []
for i in range_pos_test:
	img = np.array(cv2.imread(f'media/pos/{i}.jpg'))
	array_pos += [img]
	pass
array_pos = np.array(array_pos)

array_neg = []
for i in range_neg_test:
	img = np.array(cv2.imread(f'media/neg/{i}.jpg'))
	array_neg += [img]
	pass
array_neg = np.array(array_neg)

x_test = np.concatenate((array_pos,array_neg), axis = 0)
y_test = np.append(np.zeros(array_pos.shape[0]),np.ones(array_neg.shape[0]))
'''
y_test = np.ones((x_test.shape[0], 2))
y_test[:array_pos.shape[0], 1] = 0.
y_test[array_pos.shape[0]:, 0] = 0.
'''
x_test2 = x_test/255 - 0.5

