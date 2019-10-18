# Helper module to generate useful images for presentations

import cv2
import time
import numpy
import random
import math
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np

# a hack to wake our bus if it hangs....
import subprocess

p = subprocess.run(['i2cdetect', '-y', '1', '0x33', '0x33'])
#######################################



camera = PiCamera()
camera.resolution = (288, 368)  # start with a slightly larger image so we can crop and align later!
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(288, 368))

resolution = (240,320)


# allow the camera to warmup
time.sleep(0.1)

nmin = 0
nmax = 255
alpha1 = 0.5
alpha2 = 0.5

prevData = []
end = time.time() + 15

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    # Capture frame-by-frame
    frame = frame.array
    frame = cv2.flip(frame, 0)  # flip if neccesary

    # crop and align visible image...
    # crop image y start yend, xstart xend
    frame = frame[5:325, 10:250]

    heatmap = np.zeros((32, 24, 3), np.uint8)  # create the blank image to work from

    data = np.fromfile('/tmp/heatmap.csv', dtype=float, count=-1, sep=',')  # get the data
    if np.array_equal(data, prevData):
        print('Data stall...Probing i2c')
        p = subprocess.run(['i2cdetect', '-y', '1', '0x33', '0x33'])

    prevData = data

    index = 0
    # add to the image
    if len(data) == 768:
        for y in range(0, 32):
            for x in range(0, 24):
                val = (data[index] * 10) - 100
                if math.isnan(val):
                    val = 0
                if val > 255:
                    val = 255
                # print(index)
                # print(data)

                heatmap[y, x] = (val, val, val)

                if (y == 16) and (x == 12):
                    temp = data[index]
                index += 1
        prev_heatmap = heatmap  # save the heatmap in case we get a data miss

    else:
        print("Data miss...Loading previous thermal image")
    try:
        heatmap = prev_heatmap
    except:
        print("Previous heatmap does not exist!")

    heatmap = cv2.normalize(heatmap, None, nmin, nmax, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (240, 320), interpolation=cv2.INTER_CUBIC)

    # Display the resulting frame
    cv2.namedWindow('Thermal', cv2.WINDOW_NORMAL)
    cv2.imshow('Thermal', heatmap)

    rawCapture.truncate(0)

    res = cv2.waitKey(1)
    # print(res)

    if res == 113:  # q
        break
    if res == 97:  # a
        nmin += 10
        print(nmin)
    if res == 122:  # z
        nmin -= 10
        print(nmin)
    if res == 115:  # s
        nmax += 10
        print(nmax)
    if res == 120:  # x
        nmax -= 10
        print(nmax)

    cv2.imwrite('testing_heatmap.jpg', heatmap)
    cv2.imwrite('testing_video.jpg', frame)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    frame = cv2.filter2D(frame, -1, kernel)
    frame = cv2.addWeighted(frame, alpha1, heatmap, alpha2, 0)  # combine the images
    cv2.imwrite('testing_combo_frame.jpg', frame)

    if time.time() >= end: # stop process after 30 seconds
        break


cv2.destroyAllWindows()