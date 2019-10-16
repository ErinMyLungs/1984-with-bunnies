# simple helper module to get FPS of camera. Not easily queried by openCV
import cv2
import time

def get_fps(camera_object, camera_id_number=0, frames_to_capture=120):
    """
    reads a bunch of frames and takes the time before and after to calc fps
    :param camera_object: opened camera object
    :param camera_id_number: hardware ID number for camera
    :param frames_to_capture: number of frames to capture. Higher values take longer
    :return: estimated fps count
    """
    if not camera_object.isOpened():
        print("Camera is not opened! Attempting to start")
        camera_object = camera_object.open(camera_id_number)
        if not camera_object:
            print('Failed to open on ID, aborting function')
            raise IOError

    start = time.time()

    for i in range(frames_to_capture):
        ret, frame = camera_object.read()

    end = time.time()

    elapsed = end-start
    fps = frames_to_capture/elapsed
    print(f'total time: {elapsed} seconds')
    print()
    print(f'estimated fps: {fps}')
    return fps

if __name__ == '__main__':
    cam_num = int(input('camera id number (0 for pi, 1 for web): '))

    test_cam = cv2.VideoCapture(cam_num)
    fps = get_fps(test_cam, cam_num, 600)