import numpy as np
import cv2
import math


# Read the original image
img = cv2.resize(cv2.imread('pinza/Image__2022-05-11__16-43-02.bmp'), (1296,972))
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
edges=cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)[1]
# edges[440,:]=255
# edges[500,:]=255

cv2.imshow('Canny Edge Detection', edges)

point1=(80,np.where(edges[:,80]==255)[0][0])
point2=(1216,np.where(edges[:,1216]==255)[0][0])
cv2.circle(img, point1, 10, (0,0,255))
cv2.circle(img, point2, 10, (0,0,255))
cv2.imshow('Circles', img)

if (440 < point1[1] < 500) and (440 < point2[1] < 500) and (abs(point1[1] - point2[1]) < 20):
    print("CORRECTE")
else:
    print("INCORRECTE")
cv2.waitKey(0)
cv2.destroyAllWindows()
