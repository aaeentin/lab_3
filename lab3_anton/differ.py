import cv2
import numpy as np
from skimage.measure import compare_ssim
q = 0
scores = []
while True:
    if (q == 211): break
    before = cv2.imread(f"original/original{q}.png", cv2.IMREAD_COLOR)
    after = cv2.imread(f"decoded/decoded{q}.png", cv2.IMREAD_COLOR)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = compare_ssim(before_gray, after_gray, full=True)
    print("Image similarity", score)
    scores.append(score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(filled_after, [c], 0, (0,150,0), -1)

    for i in range(before.shape[0]):
        for j in range(before.shape[1]):
            for k in range(before.shape[2]):
                filled_after[i,j,k] = (int(filled_after[i,j,k]) + int(after[i,j,k])) / 2


    cv2.imwrite(f"differ/differ{q}.png", filled_after)
    q += 1
f=open('similarity scores.txt','w')
for ele in scores:
    f.write(str(ele)+'\n')

f.close()

