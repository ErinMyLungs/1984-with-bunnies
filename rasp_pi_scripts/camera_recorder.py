import numpy as np
import cv2

def record_videofeed(camera_object, filename, resolution=None):
    """
    Takes in camera object and records to filename until stopped with q command
    :param camera_object: cv2.VideoCapture of device
    :param filename: filename to record to
    :return: filename.avi in $PWD
    """
    if not camera_object.isOpened():
        print("Camera not opened!")
        raise ValueError
    
    if not resolution:
        # pulls native res from cam and makes tuple of ints for VideoWriter object
        resolution = (int(camera_object.get(3)), int(camera_object.get(4)))
    
    camera_object.set(3, resolution[0])
    camera_object.set(4, resolution[1])
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
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
    camera_object.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam = cv2.VideoCapture(0)
    # camera_module = cv2.VideoCapture(usePiCamera=True)
    resolution = (1280, 720)
    record_videofeed(webcam, "space_test", resolution)