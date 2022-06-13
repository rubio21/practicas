import math
from skimage import io
import os
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.transform import rescale, rotate
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import cv2
import numpy as np
from skimage.feature import blob_dog, blob_doh, blob_log




im = cv2.imread("Enumera/20220518_DW-1_t0.png", cv2.IMREAD_GRAYSCALE)
params = cv2.SimpleBlobDetector_Params()
params.filterByCircularity = True
params.maxCircularity = 0.8
params.minCircularity=0.1

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(im)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey()
# image_collection = 'Enumera/*.png'
# collection = io.imread_collection(image_collection)
#
# def filter_incomplete_blobs(blobs, im_shape):
#     h, w = im_shape
#     filtered_blobs = []
#     for blob in blobs:
#         y, x, r = blob
#
#         top = y-r
#         left = x-r
#         right = x+r
#         bottom = y+r
#
#         if(top>0) & (left>0) & (right<w) & (bottom<h):
#             filtered_blobs.append(blob)
#
#     return filtered_blobs
#
#
# for image in collection:
#     image = rgb2gray(image)
#     image = rescale(image, 0.5)
#
#     min_sigma = 15
#     max_sigma = 30
#
#     # blob_method = blob_dog
#     # blob_method = blob_log
#     blob_method = blob_doh
#
#     blobs = blob_method(image, min_sigma=min_sigma, max_sigma=max_sigma, overlap = 0, threshold=.05)
#     blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
#
#     blobs = filter_incomplete_blobs(blobs, image.shape)
#
#     sigma = 3
#     blurred_image = gaussian(image, sigma=sigma)
#     blurred_blobs = blob_method(blurred_image, min_sigma=min_sigma, max_sigma=max_sigma, overlap = 0, threshold=.05)
#     blurred_blobs[:, 2] = blurred_blobs[:, 2] * math.sqrt(2)
#
#     blurred_blobs = filter_incomplete_blobs(blurred_blobs, image.shape)
#
#     blobs_list = [blobs, blurred_blobs]
#     image_list = [image, blurred_image]
#     sequence = zip(blobs_list, image_list)
#
#     fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
#     ax = axes.ravel()
#
#     for idx, (blobs, image) in enumerate(sequence):
#         ax[idx].imshow(image, cmap = 'gray')
#         for blob in blobs:
#             y, x, r = blob
#             c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
#             ax[idx].add_patch(c)
#
#     plt.tight_layout()
#     plt.tight_layout()
#     plt.show()
#     #break