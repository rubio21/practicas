import numpy as np
import cv2 as cv
import glob
import pyrealsense2 as rs
import cv2

# Load previously saved data
with np.load('CameraParams.npz') as file:
    ret, mtx, dist, rvecs, tvecs = [file[i] for i in ('ret','mtx','dist','rvecs','tvecs')]


def draw(img, corners, imgpts):

    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)

    return img


def drawBoxes(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    # img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    # for i,j in zip(range(4),range(4,8)):
        # img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[1]),(0,255,0) , 3)
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[3]), (255,0,0), 3)
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[4]),(0,0,255),3)
    # draw top layer in red color
    # img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img



criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisBoxes = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

pipeline=rs.pipeline()
realsense_cfg=rs.config()
realsense_cfg.enable_stream(rs.stream.color, 1280,720,rs.format.rgb8, 6)
pipeline.start(realsense_cfg)

print('Test data source...')
try:
    np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
except:
    raise Exception("Can't get rgb")

while True:
    # read the current frame
    # ret, frame = cap.read()
    # if not ret:
    #     print("Unable to capture video")
    #     return
    frames=pipeline.wait_for_frames()
    frame=np.asanyarray(frames.get_color_frame().get_data())
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)

        frame = drawBoxes(frame,corners2,imgpts)
        cv.imshow('img',frame)
        k = cv.waitKey()





cv.destroyAllWindows()