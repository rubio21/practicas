import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt

namedir='Enumera'
lis = os.listdir('Enumera')

for ii in range(0,len(lis),2):
    # Read input
    color=cv2.imread(namedir + '/' + lis[ii])    # color = cv2.resize(color, (0, 0), fx=0.15, fy=0.15)
    color = cv2.GaussianBlur(color, (17,17), 0)
    cv2.imwrite('output/gaussian.png', color)
    # RGB to gray
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/gray.png', gray)
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
        # Estimate the bounding rect area:
        rectArea = boundRect[2] * boundRect[3]
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
            rect=cv2.minAreaRect(hull)
            box=np.int0(cv2.boxPoints(rect)).reshape(-1, 1, 2)
            for i in box:
                cv2.circle(inputImageCopy, (i[0], i[1]), 5, 255, 5)
            cv2.imshow('a',cv2.resize(inputImageCopy,(600,800)))
            cv2.waitKey()
