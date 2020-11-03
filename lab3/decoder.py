#import numpy as np
import cv2
import time

def apply_flow(img, flow):
    tmp = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                try:
                    tmp[int(i+flow[i,j,0]),int(j+flow[i,j,1]), k] = img[i,j,k]
                except Exception as e:
                    print(i,j,k)
                    print(img[i,j,k])
                    print(flow[i,j,0], flow[i,j,1])
    return tmp

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('decoded.avi',fourcc, 20.0, (360,288))

dts = []

cap = cv2.VideoCapture('encoded.avi')
ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
out.write(prev)
i = 0
while (True):
    ret, img = cap.read()
    if (ret == 0): break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    t1 = time.time()

    pred = apply_flow(prev, flow)

    t2 = time.time()
    dt = t2 - t1
    dts.append(dt)

    cv2.imshow('pred', pred)
    prev = img    
    out.write(pred)
    cv2.imwrite(f'decoded/decoded{i}.png', pred)
    i += 1
    out.write(img)

    ch = 0xFF & cv2.waitKey(5)
    if ch == 27:
        break
#!/usr/bin/python


f=open('times.txt','w')
for ele in dts:
    f.write(str(ele)+'\n')

f.close()

cap.release()
out.release()
cv2.destroyAllWindows()     
