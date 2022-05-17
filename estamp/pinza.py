import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the original image
img = cv2.resize(cv2.imread('pinza/ok.bmp'), (1296,972))
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

rho, theta, thresh = 2, np.pi/180, 400
lines = cv2.HoughLines(edges, rho, theta, thresh)
cv2.imshow('lines', lines)

cv2.destroyAllWindows()
