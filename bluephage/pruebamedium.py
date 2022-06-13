import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt


def line_intersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
       return
    d = (det(*line1), det(*line2))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    if 0<y<height and 0<x<width:
        intersection_points.append([x,y])
        cv2.circle(inputImageCopy, (x,y), 5, 255, 5)

namedir='Enumera'
lis = os.listdir('Enumera')

for ii in range(0,len(lis)):
    # Read input
    color=cv2.imread(namedir + '/' + lis[ii])    # color = cv2.resize(color, (0, 0), fx=0.15, fy=0.15)
    color = cv2.GaussianBlur(color, (17,17), 0)
    cv2.imwrite('output/gaussian.png', color)
    # RGB to gray
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/gray.png', gray)
    # cv2.imwrite('output/thresh.png', thresh)
    # Edge detection
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    # Save the edge detected image
    cv2.imwrite('output/edges.png', edges)
    imagecontours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Store the corners:
    cornerList = []
    inputImageCopy = color.copy()

    # Look for the outer bounding boxes (no children):
    for i, c in enumerate(imagecontours):
        # Approximate the contour to a polygon:
        contoursPoly = cv2.approxPolyDP(c, 3, True)
        # Convert the polygon to a bounding rectangle:
        boundRect = cv2.boundingRect(contoursPoly)
        # Get the bounding rect's data:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]
        # Estimate the bounding rect area:
        rectArea = rectWidth * rectHeight
        # Set a min area threshold
        minArea = 600000
        # Filter blobs by area:
        if rectArea > minArea:
            # Get the convex hull for the target contour:
            hull = cv2.convexHull(contoursPoly)
            # (Optional) Draw the hull:
            cv2.polylines(inputImageCopy, [hull], True, (0, 0, 255), 2)
            # Create image for good features to track:
            (height, width) = edges.shape[:2]
            # Black image same size as original input:
            hullImg = np.zeros((height, width), dtype=np.uint8)
            # Draw the points:
            cv2.drawContours(hullImg, [hull], 0, 255, 2)
            # cv2.imshow("hullImg", cv2.resize(hullImg,(600,800)))
            # cv2.waitKey(0)
            # rect=cv2.minAreaRect(hull)
            # box=np.int0(cv2.boxPoints(rect))
            # for i in box:
            #     cv2.circle(inputImageCopy, (i[0], i[1]), 5, 255, 5)
            # cv2.imshow('a',cv2.resize(inputImageCopy,(600,800)))
            # cv2.waitKey()

    v1, v2, h1, h2 = int(height * 0.1875), int(height * 0.84), int(width * 0.3), int(width * 0.6)
    point1 = (np.where(hullImg[v1, :] == 255)[0][0], v1)
    point2 = (np.where(hullImg[v2, :] == 255)[0][0], v2)
    point3 = (np.where(hullImg[v1, :] == 255)[0][-1], v1)
    point4 = (np.where(hullImg[v2, :] == 255)[0][-1], v2)
    point5 = (h1, np.where(hullImg[:, h1] == 255)[0][-1])
    point6 = (h2, np.where(hullImg[:, h2] == 255)[0][-1])
    points = [[point1, point2], [point3, point4], [point5, point6]]
    lines = []
    for p1, p2 in points:
        theta = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        inix = int(p1[0] + 10000 * np.cos(theta))
        iniy = int(p1[1] + 10000 * np.sin(theta))
        endpt_x = int(p1[0] - 10000 * np.cos(theta))
        endpt_y = int(p1[1] - 10000 * np.sin(theta))
        cv2.line(inputImageCopy, (inix, iniy), (endpt_x, endpt_y), 255, 2)
        lines.append([[inix, iniy], [endpt_x, endpt_y]])

    intersection_points = []
    for i in range(len(lines) - 1):
        for j in range(i, len(lines)):
            line_intersection(lines[i], lines[j])
    # print(intersection_points)
    mediox, medioy=int((intersection_points[1][0] + intersection_points[0][0]) / 2), int((intersection_points[1][1] + intersection_points[0][1]) / 2)
    cv2.circle(inputImageCopy, (mediox, medioy), 5, 255, 5)
    inix=lines[0][0][0]+mediox-intersection_points[0][0]
    endpt_x=lines[0][1][0]+mediox-intersection_points[0][0]
    punto1=(inix, lines[0][0][1])
    punto2=(endpt_x, lines[0][1][1])
    cv2.line(inputImageCopy, punto1, punto2, (255,255,0), 2)


    for i in range(inputImageCopy.shape[1]):
        x=inputImageCopy[:,i][0]
        if list(x)==[255,255,0]:
            punto_top=(i,0)

    for i in range(inputImageCopy.shape[1]):
        x=inputImageCopy[:,i][-1]
        if list(x)==[255,255,0]:
            punto_down=(i,height)

    m = (punto1[1] - punto2[1]) / (punto1[0] - punto2[0])
    b0 = punto1[1] - (m * punto1[0])
    b1 = punto2[1] - (m * punto2[0])
    b = b1

    l = np.arange(punto_down[0], punto_top[0], 0.0001)
    print(m, b)
    c=0
    for x in l:
        if (m * x + b) > 1600 or (m * x + b) < 0:
            c=1
            break
        if(hullImg[int(m*x+b),int(x)]==255):
            cv2.circle(inputImageCopy, (int(x), int(m * x + b)), 5, 255, 5)
    if c==0:
        print('tamo dentro')
        l = np.arange( punto_top[0]-10, punto_down[0],  0.001)
        for x in l:
            if (m * x + b) > 1600 or (m * x + b) < 0:
                continue
            if (hullImg[int(m * x + b), int(x)] == 255):
                cv2.circle(inputImageCopy, (int(x), int(m * x + b)), 5, 255, 5)

    iniy = lines[2][0][1] + (m * x + b) + intersection_points[0][1]
    endpt_y = lines[2][1][1] + (m * x + b) + intersection_points[1][1]
    punto1 = (lines[2][0][0], int(iniy))
    punto2 = (lines[2][1][0], int(endpt_y))
    cv2.line(inputImageCopy, punto1, punto2, (255,255,0), 2)

    cv2.imshow('color',cv2.resize(inputImageCopy,(600,800)))
    cv2.waitKey()