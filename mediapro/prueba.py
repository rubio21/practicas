import cv2
import numpy as np
import math



imgname = 'cocacola-logo.jpg'        # query image (small object)
imgname2 = 'a.png' # train image (large scene)
MIN_MATCHES = 10

def draw(img, corners, imgpts):

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)
    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)



homography = None
# matrix of camera parameters (made up but works quite well for me)
camera_parameters = np.array([[800, 0, 900], [0, 800, 450], [0, 0, 1]])
# create ORB keypoint detector
sift = cv2.SIFT_create()
# create BFMatcher object based on hamming distance
bf = cv2.BFMatcher()
# load the reference surface that will be searched in the video stream
model = cv2.imread(imgname, 0)
# Compute model keypoints and its descriptors
kp_model, des_model = sift.detectAndCompute(model,None)

frame = cv2.imread(imgname2)
# find and draw the keypoints of the frame
kp_frame, des_frame = sift.detectAndCompute(frame, None)
# match frame descriptors with model descriptors
matches = bf.match(des_model, des_frame)
# sort them in the order of their distance
# the lower the distance, the better the match
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    # differenciate between source points and destination points
    src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # compute Homography
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Draw a rectangle that marks the found model in the frame
    h, w = model.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # project corners into frame
    dst = cv2.perspectiveTransform(pts, homography)
    # connect them with lines
    frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    projection = projection_matrix(camera_parameters, homography)
    frame = render(frame, obj, projection, model, False)

cv2.imshow('aa',frame)
cv2.imwrite('found.png', frame)
cv2.waitKey()
