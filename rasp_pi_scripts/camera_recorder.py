import numpy as np
import cv2

def record_videofeed(camera_object, filename):
    """
    Takes in camera object and records to filename until stopped with q command
    :param camera_object: cv2.VideoCapture of device
    :param filename: filename to record to
    :return: filename.avi in $PWD
    """
    resolution = (int(camera_object.get(3)), int(camera_object.get(4))) #pulls native resolution for the cam
    fourcc = cv2.VideoWriter_fourcc(*'X264') # this is currently broken
    output = cv2.VideoWriter(f'{filename}.avi', fourcc, 30.0, resolution) # output name, encoding, FPS, resolution tuple


    while(camera_object.isOpened()):
        # Capture frame-by-frame
        ret, frame = camera_object.read()

        if ret == True:
            output.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    # When everything done, release the capture and output
    webcam.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam = cv2.VideoCapture(0)
    # camera_module = cv2.VideoCapture(usePiCamera=True)
    record_videofeed(webcam, "output_test")