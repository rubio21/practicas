import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil

# Load the images
name = "20220211_100402_A5EF_567_3"

img1 = cv2.imread('homografias/' + name + '.png',0)

# Calculate the histograms, and normalize them
hist_img1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
plt.plot(hist_img1)
plt.show()
carpeta = os.listdir('homografias')
for foto in carpeta:
    x = np.float32(np.zeros((256, 1)))
    y = hist_img1[90:214]
    img2 = cv2.cvtColor(cv2.imread('homografias/' + foto), cv2.COLOR_BGR2GRAY)
    hist_img2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    HISTCMP_BHATTACHARYYA=[]
    HISTCMP_INTERSECT=[]
    HISTCMP_CHISQR=[]
    HISTCMP_CORREL=[]
    for i in range(0,133,4):
        # print(i)
        if i>1:
            x[0:i-1]=0
        x[i:124+i] = y
        plt.plot(x)
        plt.plot(hist_img2)
        plt.show()
        HISTCMP_BHATTACHARYYA.append(cv2.compareHist(x, hist_img2, cv2.HISTCMP_BHATTACHARYYA))
        HISTCMP_INTERSECT.append(cv2.compareHist(x, hist_img2, cv2.HISTCMP_INTERSECT))
        HISTCMP_CHISQR.append(cv2.compareHist(x, hist_img2, cv2.HISTCMP_CHISQR))
        HISTCMP_CORREL.append(cv2.compareHist(x, hist_img2, cv2.HISTCMP_CORREL))

    if (min(HISTCMP_BHATTACHARYYA) <= 0.36 and max(HISTCMP_INTERSECT) >= 35000 and max(HISTCMP_CORREL)>=0.52):

        print(foto)
        print("HISTCMP_BHATTACHARYYA: ", min(HISTCMP_BHATTACHARYYA))
        print("HISTCMP_INTERSECT: ", max(HISTCMP_INTERSECT))
        print("HISTCMP_CHISQR: ", min(HISTCMP_CHISQR))
        print("HISTCMP_CORREL: ", max(HISTCMP_CORREL))
        print("-------------------------")
        # shutil.copyfile('homografias/' + foto, 'bien/' + foto)

    # else:
        # shutil.copyfile('homografias/' + foto, 'mal/' + foto)

