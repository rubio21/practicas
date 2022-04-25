import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2


color_raw = o3d.io.read_image("a.png")
depth_raw = o3d.io.read_image("KinectScreenshot-21-04-2022-15-35-39-945_d.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0)
print(rgbd_image)

# Plot the images
plt.subplot(1, 2, 1)
plt.title('Grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_image.depth)
plt.show()