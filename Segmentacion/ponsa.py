import math
import os
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_doh, blob_log
import cv2

path = 'apples2'
collection = os.listdir(path)

def filter_incomplete_blobs(blobs, im_shape):
    h, w = im_shape
    filtered_blobs = []
    for blob in blobs:
        y, x, r = blob
        top = y-r
        left = x-r
        right = x+r
        bottom = y+r
        if(top>0) & (left>0) & (right<w) & (bottom<h):
            filtered_blobs.append(blob)
    return filtered_blobs


for image in collection:
    image = cv2.imread(path + '/' + image)
    image = rgb2gray(image)
    image = rescale(image, 0.25)

    min_sigma = 15
    max_sigma = 30

    blob_method = blob_dog
    # blob_method = blob_log

    blobs = blob_method(image, min_sigma=min_sigma, max_sigma=max_sigma, overlap = 0, threshold=.05)
    blobs[:, 2] = blobs[:, 2] * math.sqrt(2)

    blobs = filter_incomplete_blobs(blobs, image.shape)

    sigma = 3
    blurred_image = gaussian(image, sigma=sigma)
    blurred_blobs = blob_method(blurred_image, min_sigma=min_sigma, max_sigma=max_sigma, overlap = 0, threshold=.05)
    blurred_blobs[:, 2] = blurred_blobs[:, 2] * math.sqrt(2)

    blurred_blobs = filter_incomplete_blobs(blurred_blobs, image.shape)

    blobs_list = [blobs, blurred_blobs]
    image_list = [image, blurred_image]
    sequence = zip(blobs_list, image_list)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, image) in enumerate(sequence):
        ax[idx].imshow(image, cmap = 'gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax[idx].add_patch(c)

    plt.tight_layout()
    plt.tight_layout()
    plt.show()

