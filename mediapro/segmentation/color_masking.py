import cv2
import numpy as np
from matplotlib import pyplot as plt

path = '../c.png'
img = cv2.imread(path)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

light_green = (40, 40, 40)
dark_green = (70, 255, 255)
mask = cv2.inRange(hsv_img, light_green, dark_green)

result = cv2.bitwise_and(img, img, mask=mask)
result[result != 0] = 255

kernel = np.ones((10, 10), np.uint8)
img_dilation = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)

kernel = np.ones((10, 10), np.uint8)
img_dilation = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel, iterations=10)

plt.subplot(211),plt.imshow(result)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(img_dilation, 'gray')
# plt.imsave(r'thresh.png',thresh)
plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
# cv2.imshow('r', cv2.resize(img_dilation, (1400, 788)))
# cv2.waitKey()
