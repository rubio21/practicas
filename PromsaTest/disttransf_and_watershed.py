import numpy as np
import cv2

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(opening,(x,y),15,100,-1)
        mouseX,mouseY = x,y

def draw_line(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # cv2.circle(img,(x,y),15,100,-1)
        ar.append((x,y))
        print(x,y, len(ar))
        if len(ar)==2:
            cv2.line(opening, ar[0], ar[1], 100, 5)
            ar.clear()

img = cv2.resize(cv2.imread('Arros50mm.png'),(512,512))
cv2.imshow('src', img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh=cv2.bitwise_not(thresh)
cv2.imshow('thresh', thresh)

# noise removal
kernel = np.ones((6,6),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imshow('opening', opening)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imshow('sure_bg', sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
# cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(),255,0)



# ar=[]
# cv2.namedWindow('click')
# cv2.setMouseCallback('click',draw_line)
# while(1):
#     cv2.imshow('click',opening)
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27:
#         break
#
# sure_fg = np.where(opening != 100, 0, 255).astype(np.uint8)
cv2.imshow('sure_fg', sure_fg)
# cv2.waitKey()


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('unknown', unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0


markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
norm_arr = (markers - np.min(markers)) / (np.max(markers) - np.min(markers))
cv2.imshow('norm_arr', norm_arr)
cv2.imshow('img', img)
cv2.waitKey()