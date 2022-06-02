import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    imgpts = np.int32(imgpts)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw2(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img





def plannar_matching(img1,img2, show = True):
    MIN_MATCH_COUNT = 10
    
    # Initiate SIFT detector
    sift = cv2.ORB_create()#xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # #match keypoints
    # print(des1)
    # matches = flann.knnMatch(des1,des2,k=2)

    # Create brute force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)    
    good = sorted(matches, key = lambda x:x.distance)
    # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.9*n.distance:
    #         good.append(m)
            
         
    if len(good)>MIN_MATCH_COUNT:
        success = True
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #find homography + RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
        matchesMask = mask.ravel().tolist()

        
        if show:    
            allMask = np.ones(len(matchesMask)).tolist()
            draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = allMask, # draw only inliers
                               flags = 2)
            
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)#,**draw_params)

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)
            
            img3 = cv2.drawMatches(img3,kp1,img2,kp2,good,None,**draw_params)


            plt.figure()
            plt.imshow(img3, 'gray')
            plt.show()  
        
        
        #Estmated intrinsec camera matrix
        size = img1.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        mtx = np.array([[focal_length, 0, center[1]],
                          [0, focal_length, center[0]],
                          [0, 0, 1]], dtype = "double"
                          )
        # mtx = np.array([[616.509, 0, 320.977],
        #                   [0, 616.509, 245.527],
        #                   [0, 0, 1]], dtype = "double"
        #                   )
        
        dist = np.zeros((4,1))
        
        
        
        origin = np.float32([[0,0,0]]).reshape(-1,3)
        axis = np.float32([[500,0,0], [0,500,0], [0,0,-500]]).reshape(-1,3)
        
        #Good matches to estimate pose
        goodMask = [item for ii,item in enumerate(good) if matchesMask[ii] == 1]

        if len(goodMask)>MIN_MATCH_COUNT:
            src_matches = np.float32([ kp1[m.queryIdx].pt for m in goodMask ]).reshape(-1,2)
            src_matches[:,0] = src_matches[:,0] - 20#src_matches[0,0]
            src_matches[:,1] = src_matches[:,1] - 30#src_matches[0,1]
            
            dst_matches = np.float32([ kp2[m.trainIdx].pt for m in goodMask ]).reshape(-1,2)
            dst_matches[:,0] = dst_matches[:,0] #- dst_matches[10,0]
            dst_matches[:,1] = dst_matches[:,1] #- dst_matches[10,1]
            
            #transform 2d coordinates to 3d
            src_matches_3d = np.zeros((src_matches.shape[0], 3))
            src_matches_3d[:,:2] = src_matches
            
            
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv2.solvePnP(src_matches_3d, dst_matches, mtx, dist)
            
            if show:
                # project 3D points to image plane representing axis
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                origin_trans, jac = cv2.projectPoints(origin, rvecs, tvecs, mtx, dist)
                
                img = draw(col_img,origin_trans.astype(int),imgpts)
                img = cv2.resize(img, (640, 480))
                cv2.imshow('img1',img)
                            
                # cv2.imshow('err1',img2)
                cv2.waitKey(0)
                
            # We hide the text found to find another one in the image in case
            # there are more pieces.
            im_dst = np.logical_not(cv2.warpPerspective(blank, H, (sy,sx)))
            img2 = img2*im_dst
            return success, H, rvecs, tvecs, img2
                
        else:
            print('Not enough RANSAC matches')
            return success, None, None, None, img2
    
    else:
        if show:    

            plt.figure()
            plt.imshow(img2, 'gray')
            plt.show()  
            
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        success = False
        
        return success, None, None, None, img2
    
    
col_img = cv2.imread('img.jpg')

img2 = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
sx,sy = img2.shape

                   
img1 = cv2.imread('PapaPitufo.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
blank = np.ones_like(img1)




success, H, rvecs, tvecs, img2 = plannar_matching(img1,img2)

