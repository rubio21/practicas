import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

i=0
for j in os.listdir('pinza'):
    # Read the original image
    img = cv2.resize(cv2.imread('pinza/' +j), (1296,972))
    ## Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    edges=cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)[1]

    point1=(80,np.where(edges[:,80]==255)[0][0])
    point2=(1216,np.where(edges[:,1216]==255)[0][0])
    # cv2.circle(img, point1, 10, (0,0,255))
    # cv2.circle(img, point2, 10, (0,0,255))

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    if (440 < point1[1] < 500) and (440 < point2[1] < 500) and (abs(point1[1] - point2[1]) < 20):
        plt.axline((point1[0], point1[1]), (point2[0], point2[1]), color='g')
    else:
        plt.axline((point1[0], point1[1]), (point2[0], point2[1]), color='r')

    plt.savefig('pinza2/' +str(i)+ '.png')
    i+=1

cv2.destroyAllWindows()
