import matplotlib as plt
import numpy as np
import cv2
path = '../captura.png'
img = cv2.imread(path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
twoDimage = img.reshape((-1, 3))
twoDimage = np.float32(twoDimage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
attempts = 10

ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
cv2.imshow('r',cv2.resize(result_image,(1400, 788)))
cv2.waitKey()