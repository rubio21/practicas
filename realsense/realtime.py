import cv2
import numpy as np
import pyrealsense2 as rs
import math

ESC = 27
WINDOW_NAME = 'REALSENSE TEST'
point=(400,300)

def show_distance(event, x, y , args, params):
    global point
    point=(x,y)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, show_distance)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.depth)
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        image_8_bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(image_8_bit, cv2.COLORMAP_JET)

        depth_colormap_shape = depth_colormap.shape
        color_image_shape = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_shape != color_image_shape:
            h, w = depth_colormap.shape
            color_image = cv2.resize(color_image, dsize=(w, h), interpolation=cv2.INTER_AREA)

        cv2.circle(color_image, point, 4, (0, 0, 255))
        cv2.circle(depth_colormap, point, 4, (0, 0, 255))

        depth = depth_frame.get_distance(point[0], point[1])
        distance = depth_image[point[1], point[0]]

        cv2.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME, images)


        if cv2.waitKey(1) == ESC:
            cv2.destroyAllWindows()
            quit()
finally:
    # Stop streaming
    pipeline.stop()

