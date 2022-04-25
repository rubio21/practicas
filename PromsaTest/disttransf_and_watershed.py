import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

img = cv2.resize(cv2.imread('arros02.bmp'),(512,512))
cv2.imshow('src', img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh=cv2.bitwise_not(thresh)
cv2.imshow('thresh', thresh)

# noise removal
kernel = np.ones((6,6),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imshow('opening', opening)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
imS = cv2.resize(sure_bg, (512, 512))
cv2.imshow('sure_bg', imS)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
cv2.imshow('sure_fg', sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0


markers = cv2.watershed(img, markers)
img[markers == -1] = [255,0,0]
norm_arr = (markers - np.min(markers)) / (np.max(markers) - np.min(markers))
cv2.imshow('norm_arr', norm_arr)
cv2.imshow('img', img)
cv2.waitKey()


