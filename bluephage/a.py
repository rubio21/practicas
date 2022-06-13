import os
import cv2
import cv2 as cv
import numpy as np


def esquinas(img):
    color = cv2.GaussianBlur(img, (17,17), 0)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    imagecontours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
            cv2.polylines(img, [hull], True, (0, 0, 255), 2)
            # Create image for good features to track:
            (height, width) = edges.shape[:2]
            # Black image same size as original input:
            hullImg = np.zeros((height, width), dtype=np.uint8)
            # Draw the points:
            cv2.drawContours(hullImg, [hull], 0, 255, 2)
            # cv2.imshow("hullImg", cv2.resize(hullImg,(600,800)))
            # cv2.waitKey(0)
            rect = cv2.minAreaRect(hull)
            box = np.int0(cv2.boxPoints(rect))


            cv2.polylines(img, [box], True, (0, 255, 0), 4)
            box=np.array(sorted(box, key=lambda x: sum(x)))
            for i in box:
                cv2.circle(img, (i[0], i[1]), 5, 255, 8)

            cv2.imshow('a',cv2.resize(img,(600,800)))
            cv2.waitKey()

            box = box.reshape(-1, 1, 2)

            return box
    return None

namedir='Enumera'
lis = os.listdir('Enumera')
for ii in range(0,len(lis)-1,2):
    print(ii)
    color_1 = cv.imread(namedir + '/' + lis[ii])
    img1 = cv.cvtColor(color_1, cv.COLOR_BGR2GRAY)
    color_2 = cv.imread(namedir + '/' + lis[ii + 1])
    img2 = cv.cvtColor(color_2, cv.COLOR_BGR2GRAY)
    esquinas_1=esquinas(color_1)
    esquinas_2=esquinas(color_2)

    sx, sy = img2.shape
    # blank = np.ones_like(img1)
    # sift = cv.SIFT_create()
    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    #
    # # Match keypoints in both images
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    #
    # # Keep good matches: calculate distinctive image features
    # matchesMask = [[0, 0] for i in range(len(matches))]
    # good = []
    # pts1 = []
    # pts2 = []
    #
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.2*n.distance:
    #         # Keep this keypoint pair
    #         matchesMask[i] = [1, 0]
    #         good.append(m)
    #         pts2.append(kp2[m.trainIdx].pt)
    #         pts1.append(kp1[m.queryIdx].pt)
    #
    # # Draw the keypoint matches between both pictures
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=cv.DrawMatchesFlags_DEFAULT)
    #
    # keypoint_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # # cv.imshow("Keypoint matches", keypoint_matches)
    # # cv2.imwrite('kpmatches.jpg', keypoint_matches)
    #
    # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #
    # src_pts=np.concatenate((esquinas_1, src_pts), axis=0)
    # dst_pts=np.concatenate((esquinas_1, dst_pts), axis=0)

    # find homography + RANSAC
    H, mask = cv2.findHomography(esquinas_1, esquinas_2)
    im_out = cv2.warpPerspective(color_2, H, (sy,sx))
    cv2.imwrite(namedir + '/a/' + lis[ii+1][:-4] +'_.png', im_out)

