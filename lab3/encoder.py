import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('encoded.avi',fourcc, 20.0, (360,288))

cap = cv2.VideoCapture('slow.avi')
i = 0
while (True):
    ret, img = cap.read()
    if (ret == 0): break
    if (i % 2 == 0 ):
        out.write(img)
    else:
    	cv2.imwrite(f'original/original{int(i/2)}.png', img)
    i += 1

cap.release()
out.release()           


