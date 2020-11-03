import numpy as np
import cv2 as cv

def apply_flow(img, p0, p1):
	tmp = img
	try:
		#print (p0, p1)
		for i, j in p0, p1:
			for k in range(3):
				tmp[int(j[0][0]), int(j[0][1]),k] = img[int(i[0][0]), int(i[0][1]),k]
	except: pass
	return tmp

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('decoded.avi',fourcc, 20.0, (360,288))

cap = cv.VideoCapture('slow.avi')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
											 qualityLevel = 0.2,
											 minDistance = 10,
											 blockSize = 20)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
									maxLevel = 2,
									criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, prev = cap.read()
old_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
i = 0
while(1):
		ret,frame = cap.read()
		if ret == 0: break
		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		#try:
		# calculate optical flow
		p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
		# Select good points
		pred = apply_flow(prev, p0, p1)
		prev =  frame

		out.write(pred)
		cv.imwrite(f'decoded/decoded{i}.png', pred)

		#cv.imshow('frame',img)
		k = cv.waitKey(30) & 0xff
		if k == 27:
				break
		# Now update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = p1
		'''
		except: 
			out.write(frame)
			cv.imwrite(f'decoded/decoded{i}.png', frame)
			p0 = cv.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
			old_gray = frame_gray.copy()
			print('###')
			'''
		i += 1