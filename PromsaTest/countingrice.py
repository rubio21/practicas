import cv2 as cv
import numpy as np

# load input images for demonstration
input_rice = cv.resize(cv.imread("arros02.bmp", cv.IMREAD_GRAYSCALE),(512,512))

# local adaptive thresholding - computes local threshold based on given window size
ret, output_adapthresh = cv.threshold(input_rice,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
output_adapthresh=cv.bitwise_not(output_adapthresh)

# output_adapthresh = cv.adaptiveThreshold (input_rice, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -20.0)
cv.imshow("Adaptive Thresholding", output_adapthresh)
#cv.imwrite('rice_adapthresh.png', output_adapthresh)

# morphologial erosion - cleaning up binary images
kernel = np.ones((3,3),np.uint8)
output_erosion = cv.erode(output_adapthresh, kernel)
cv.imshow("Morphological Erosion", output_erosion)
#cv.imwrite('rice_erosion.png', output_erosion)


# Contours - Computes polygonal contour boundary of foreground objects
# apply connected components on clean binary image
contours, _ = cv.findContours(output_erosion, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
output_contour = cv.cvtColor(input_rice, cv.COLOR_GRAY2BGR)
cv.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
print("Number of detected contours", len(contours))
cv.imshow("Contours", output_contour)
cv.imwrite('rice_contours.png', output_contour)

# wait for key press
cv.waitKey(0)