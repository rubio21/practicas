import cv2 as cv
import numpy as np

# load input images for demonstration
input_rice = cv.resize(cv.imread("arros02.bmp", cv.IMREAD_GRAYSCALE),(512,512))

ret, thresh = cv.threshold(input_rice,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
thresh=cv.bitwise_not(thresh)
cv.imshow("Adaptive Thresholding", thresh)

# morphologial erosion - cleaning up
kernel = np.ones((3,3),np.uint8)
output_erosion = cv.erode(thresh, kernel)
cv.imshow("Morphological Erosion", output_erosion)

# Contours - Computes polygonal contour boundary of foreground objects apply connected components on clean binary image
contours, _ = cv.findContours(output_erosion, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
output_contour = cv.cvtColor(input_rice, cv.COLOR_GRAY2BGR)
cv.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
print("Number of detected contours", len(contours))
cv.imshow("Contours", output_contour)
cv.imwrite('rice_contours.png', output_contour)

# wait for key press
cv.waitKey(0)