# Helper module to generate useful images for presentations

import cv2
import time
import numpy as np

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

alpha1 = 0.5
alpha2 = 0.5

count = 1

heatmap_vid = cv2.VideoCapture('data/example_heatmap.avi')
raw_vid = cv2.VideoCapture('data/example_video.avi')

heat_success, heat_frame = heatmap_vid.read()
vid_success, vid_frame = raw_vid.read()

while heat_success and vid_success:
    sharpened_vid = cv2.filter2D(vid_frame, -1, kernel)
    combo_frame = cv2.addWeighted(sharpened_vid, alpha1, heat_frame, alpha2, 0)  # combine the images

    cv2.imwrite(f'testing_heatmap{count}.jpg', heat_frame)
    cv2.imwrite(f'testing_video{count}.jpg', vid_frame)
    cv2.imwrite(f'testing_combo_frame{count}.jpg', combo_frame)

    count += 1
    heat_success, heat_frame = heatmap_vid.read()
    vid_success, vid_frame = raw_vid.read()

heatmap_vid.release()
raw_vid.release()