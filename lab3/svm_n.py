import cv2, numpy as np 

########    svm training    ###########

kmeansTrainer = cv2.BOWKMeansTrainer(5)
train_pos = []
train_neg = []

sift = cv2.SIFT_create(nfeatures=100)

range_pos = range(1, 50)
range_neg = range(1, 50)

img = cv2.imread(f"media/nikolay/frame0.jpg")
kp, des = sift.detectAndCompute(img, None)
des = np.float32(des)
kmeansTrainer.add(des)
train_pos = [(img, kp, des)]

for i in range_pos:
	try:
		img = cv2.imread(f"media/nikolay/frame{i}.jpg")
		kp, des = sift.detectAndCompute(img, None)
		des = np.float32(des)
		kmeansTrainer.add(des)
		train_pos += [(img, kp, des)]
	except: pass

img = cv2.imread(f"media/nikolay/noframe0.jpg")
kp, des = sift.detectAndCompute(img, None)
des = np.float32(des)
kmeansTrainer.add(des)
train_neg = [(img, kp, des)]

for i in range_neg:
	try:

		img = cv2.imread(f"media/nikolay/noframe{i}.jpg")
		kp, des = sift.detectAndCompute(img, None)
		des = np.float32(des)
		kmeansTrainer.add(des)
		train_neg += [(img, kp, des)]
	except: pass

vocabulary = kmeansTrainer.cluster()
extractor = cv2.BOWImgDescriptorExtractor(cv2.SIFT_create(),cv2.BFMatcher());
extractor.setVocabulary(vocabulary);


i = train_pos[0]
tr_ps = [extractor.compute(i[0], i[1], i[2])[0]]
i = train_neg[0]
tr_ng = [extractor.compute(i[0], i[1], i[2])[0]]

for i in train_pos[1:]:
	tr_ps += [extractor.compute(i[0], i[1], i[2])[0]]
	pass
for i in train_neg[1:]:
	tr_ng += [extractor.compute(i[0], i[1], i[2])[0]]
	pass
tr_ps = np.array(tr_ps)
tr_ng = np.array(tr_ng)

trainingData = np.concatenate((tr_ps,tr_ng), axis = 0).astype(np.float32)
labels = np.append(np.ones(tr_ps.shape[0]),-np.ones(tr_ng.shape[0])).astype(int)


svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)



########    svm testing    ###########

'''
range_pos = range(51, 213)
range_neg = range(51, 64)
img = cv2.imread(f"media/nikolay/frame50.jpg")
kp, des = sift.detectAndCompute(img, None)
des = np.float32(des)
test_pos = [(img, kp, des)]

for i in range_pos:
    try:
        img = cv2.imread(f"media/nikolay/frame{i}.jpg")
        kp, des = sift.detectAndCompute(img, None)
        if (des is None): continue
        des = np.float32(des)
        test_pos += [(img, kp, des)]
    except: pass

img = cv2.imread(f"media/nikolay/noframe50.jpg")
print(img)
kp, des = sift.detectAndCompute(img, None)
des = np.float32(des)
test_neg = [(img, kp, des)]

for i in range_neg:
    try:

        img = cv2.imread(f"media/nikolay/noframe{i}.jpg")
        kp, des = sift.detectAndCompute(img, None)
        if (des is None): continue
        des = np.float32(des)
        test_neg += [(img, kp, des)]
    except: pass


i = test_pos[0]
ts_ps = [extractor.compute(i[0], i[1], i[2])[0]]

i = test_neg[0]
ts_ng = [extractor.compute(i[0], i[1], i[2])[0]]

for i in test_pos[1:]:
    try:
        ts_ps += [extractor.compute(i[0], i[1], i[2])[0]]
    except: pass
for i in test_neg[1:]:
    try:
        ts_ng += [extractor.compute(i[0], i[1], i[2])[0]]
    except: pass


ts_ps = np.array(ts_ps)
ts_ng = np.array(ts_ng)
testData = np.concatenate((ts_ps,ts_ng), axis = 0).astype(np.float32)

test_labels = np.append(np.ones(ts_ps.shape[0]), -np.ones(ts_ng.shape[0])).astype(int)

predicts = []

for x in testData:
    x = x.reshape((1, x.shape[0]))
    predicts += [svm.predict(x)[1][0][0]]
    pass

predicts = np.array(predicts).astype(int)
print(predicts)
print(test_labels)
err = predicts - test_labels

false_neg = 0
false_pos = 0
for i in err:
    if (i == -2 ): false_neg += 1
    if (i == 2): false_pos += 1

f=open('falses.txt','w')
f.write(f'number of false negatives = {false_neg}' + '\n')
f.write(f'number of false positives = {false_pos}' + '\n')
f.close()

print(false_neg, false_pos)
'''

############ video testing ##################

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('classified.avi',fourcc, 20.0, (1080,1920))

cap = cv2.VideoCapture('media/nikolay/norezinka.mp4')

i = 0
while (True):
    ret, img = cap.read()
    if (ret == 0): break
    try:
        kp, des = sift.detectAndCompute(img, None)
        if (des is None): continue
        des = np.float32(des)
        word = extractor.compute(img, kp, des)
        #print(word.shape)
        if svm.predict(word)[1][0][0] == 1:
            img = cv2.putText(img, 'Rezinka', (540, 960), cv2.FONT_HERSHEY_SIMPLEX,\
                1, (255,0,0), 3, cv2.LINE_AA)
            cv2.imwrite(f"media/nikolay/noframe_cl/frame{i}.jpg", img)    
        else:
            img = cv2.putText(img, 'ne Rezinka', (540, 960), cv2.FONT_HERSHEY_SIMPLEX,\
                1, (255,0,0), 3, cv2.LINE_AA)
            cv2.imwrite(f"media/nikolay/noframe_cl/frame{i}.jpg", img)    
        out.write(img)
    except: 
        cv2.imwrite(f"media/nikolay/noframe_cl/frame{i}.jpg", img)   
        out.write(img)


    i += 1

cap.release()
out.release()

out = cv2.VideoWriter('classified2.avi',fourcc, 20.0, (768,1024))

cap = cv2.VideoCapture('media/nikolay/rezinka.mp4')

i = 0
while (True):
    ret, img = cap.read()
    if (ret == 0): break
    try:
        kp, des = sift.detectAndCompute(img, None)
        if (des is None): continue
        des = np.float32(des)
        word = extractor.compute(img, kp, des)
        #print(word.shape)
        if svm.predict(word)[1][0][0] == 1:
            img = cv2.putText(img, 'Rezinka', (393, 512), cv2.FONT_HERSHEY_SIMPLEX,\
                1, (255,0,0), 3, cv2.LINE_AA)
            cv2.imwrite(f"media/nikolay/frame_cl/frame{i}.jpg", img)    
        else:
            img = cv2.putText(img, 'ne Rezinka', (393, 512), cv2.FONT_HERSHEY_SIMPLEX,\
                1, (255,0,0), 3, cv2.LINE_AA)
            cv2.imwrite(f"media/nikolay/frame_cl/frame{i}.jpg", img)    
        out.write(img)
    except: 
        cv2.imwrite(f"media/nikolay/frame_cl/frame{i}.jpg", img)   
        out.write(img)


    i += 1

cap.release()
out.release()


















'''
range_pos = range(213)
range_neg = range(37)
img = cv2.imread(f"media/nikolay/frame/frame50.jpg")
kp, des = sift.detectAndCompute(img, None)
des = np.float32(des)
test_pos = [(img, kp, des)]

for i in range_pos:
    try:
        img = cv2.imread(f"media/nikolay/frame/frame{i}.jpg")
        kp, des = sift.detectAndCompute(img, None)
        if (des is None): continue
        des = np.float32(des)
        test_pos += [(img, kp, des)]
    except: pass

img = cv2.imread(f"media/nikolay/noframe/noframe0.jpg")
print(img)
kp, des = sift.detectAndCompute(img, None)
des = np.float32(des)
test_neg = [(img, kp, des)]

for i in range_neg:
    try:

        img = cv2.imread(f"media/nikolay/noframe/noframe{i}.jpg")
        kp, des = sift.detectAndCompute(img, None)
        if (des is None): continue
        des = np.float32(des)
        test_neg += [(img, kp, des)]
    except: pass


i = test_pos[0]
ts_ps = [extractor.compute(i[0], i[1], i[2])[0]]

i = test_neg[0]
ts_ng = [extractor.compute(i[0], i[1], i[2])[0]]

for i in test_pos[1:]:
    try:
        ts_ps += [extractor.compute(i[0], i[1], i[2])[0]]
    except: pass
for i in test_neg[1:]:
    try:
        ts_ng += [extractor.compute(i[0], i[1], i[2])[0]]
    except: pass


ts_ps = np.array(ts_ps)
ts_ng = np.array(ts_ng)
testData = np.concatenate((ts_ps,ts_ng), axis = 0).astype(np.float32)

test_labels = np.append(np.ones(ts_ps.shape[0]), -np.ones(ts_ng.shape[0])).astype(int)

predicts = []

for x in testData:
    x = x.reshape((1, x.shape[0]))
    predicts += [svm.predict(x)[1][0][0]]
    pass

predicts = np.array(predicts).astype(int)
print(predicts)
print(test_labels)
err = predicts - test_labels

false_neg = 0
false_pos = 0
for i in err:
    if (i == -2 ): false_neg += 1
    if (i == 2): false_pos += 1

f=open('falses.txt','w')
f.write(f'number of false negatives = {false_neg}' + '\n')
f.write(f'number of false positives = {false_pos}' + '\n')
f.close()

print(false_neg, false_pos)

'''
