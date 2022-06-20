import os
import cv2
import numpy as np

def process_image(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    light_green = (40, 40, 40)
    dark_green = (70, 255, 255)
    mask = cv2.inRange(hsv_img, light_green, dark_green)

    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result[result != 0] = 255

    kernel = np.ones((10, 10), np.uint8)
    img_dilation = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = np.ones((10, 10), np.uint8)
    img_dilation = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel, iterations=10)
    return img_dilation


carpeta='img'
c=0
for file in os.listdir(carpeta):
    # os.rename(carpeta+'/'+file, 'cesped/'+str(c)+'.png')
    # c+=1
    img = process_image(cv2.imread(carpeta+'/'+file))
    cv2.imwrite('bin/'+file, img)
    # cv2.imshow('a',img)
    # cv2.waitKey()