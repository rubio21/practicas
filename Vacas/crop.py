import cv2
import numpy as np
import os

for carpeta in os.listdir('classified_cow_tags'):
    for foto in os.listdir('classified_cow_tags/'+carpeta):
        img = cv2.imread('classified_cow_tags/'+carpeta+ '/' +foto)
        # Cropping an image
        cropped_image = img[78:408, 81:576]
        # Display cropped image
        # cv2.imshow("cropped", cropped_image)
        # Save the cropped image
        cv2.imwrite('eigentags/' +carpeta +'/' +foto, cropped_image)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
