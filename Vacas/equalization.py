import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil


path = "malManual/20220211_093802_EE22_405_2.png"

for path in os.listdir('oscuras'):
    img = cv2.imread('oscuras/'+path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # or convert


    cv2.imshow('image',img)

    equ = cv2.equalizeHist(img)

    cv2.imshow('equ.png',equ)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite('oscuras/' + path[:-4] + '_equ.png', equ)
    # hist,bins = np.histogram(img.flatten(),256,[0,256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()



    # hist,bins = np.histogram(equ.flatten(),256,[0,256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(equ.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()
