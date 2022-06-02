import cv2
import glob
import numpy as np
# Read Image

def draw(img, corners, imgpts):
    p1 = (int(corners[0]), int(corners[1]))
    img = cv2.line(img, p1, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 5)
    img = cv2.line(img, p1, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 5)
    img = cv2.line(img, p1, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 5)
    return img


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('../calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


#2D image points. If you change the image, you need to change vector


image_points = np.array([
                        (446, 775), #1
                        (629, 601), #2
                        (205, 640), #4
                        (425, 480), #5
                        (653, 326), #6
                        (394, 251), #7
                        (147, 356) #8
                        ], dtype="double")

# image_points = np.array([
#                         (359, 359), #yellow
#                         (166, 246), #blue
#                         (360, 582), #red
#                         (555, 249), #green
#                         (358, 135), #pink
#                         (167, 472), #orange
#                         (555, 472) #white
#                          ], dtype="double")

alpha = 56.0
# 3D model points.
model_points = np.array([(0, 0, 0), #1
                         (alpha, 0, 0), #2
                         #(alpha, alpha, 0), #3
                         (0, alpha, 0), #4
                         (0, 0, alpha), #5
                         (alpha, 0, alpha), #6
                         (alpha, alpha, alpha), #7
                         (0, alpha,alpha) #8
                         ], dtype="double")

# alpha = 5
axis = np.float32([[alpha,0,0], [0,alpha,0], [0,0,alpha]]).reshape(-1,3)

im = cv2.imread('../rubik.jpg')

print ("Camera Matrix :\n {0}".format(mtx))

# Camera internals
size = im.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
mtx = np.array([[focal_length, 0, center[1]],
                  [0, focal_length, center[0]],
                  [0, 0, 1]], dtype = "double"
                  )
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

print ("Camera Matrix Approximation :\n {0}".format(mtx))
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)#cv2.SOLVEPNP_EPNP)
# success, rotation_vector, translation_vector,reprojerror = cv2.solvePnPGeneric(model_points, image_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
# success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, mtx, dist)
print ("Rotation Vector:\n {0}".format(rotation_vector))
print ("Rotation Vector in degrees:\n {0}".format(rotation_vector*180/np.pi))

print ("Translation Vector:\n {0}".format(translation_vector))

# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose

(image_points2D, jacobian) = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)

# for p in image_points:
#     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

# p1 = ( int(image_points[4][0]), int(image_points[4][1]))
# for point in image_points2D:
#     p2 = ( int(point[0][0]), int(point[0][1]))
#     print(p2)
#     cv2.line(im, p1, p2, (255,0,0), 2)
# # Display image
# cv2.imshow("Output", im)
# cv2.waitKey(0)

# (imgpts, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, mtx, dist)

for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,255,0), -1)

im = draw(im,image_points[0],image_points2D)

(image_points2D, jacobian) = cv2.projectPoints(model_points, rotation_vector, translation_vector, mtx, dist)
for p in image_points2D:
    cv2.circle(im, (int(p[0][0]), int(p[0][1])), 3, (0,0,255), -1)
    
#im = draw(im,(330.0,435.0),image_points2D)
cv2.imshow('img',im)
k = cv2.waitKey(0) & 0xff
        

