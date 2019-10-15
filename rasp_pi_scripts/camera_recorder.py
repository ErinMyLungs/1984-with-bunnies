# module to connect to and record from video camera
import numpy as np
import cv2

def cam_checker(camera_object):
    """
    basic check if camera_object is opened
    :param camera_object: cv2.VideoCapture device
    """
    if not camera_object.isOpened():
        print("Camera not opened!")
        raise ValueError
    else:
        return True
    
def resolution_setter(camera_object, resolution):
    """
    Sets resolution from argument or gets default camera resolution
    :param camera_object: cv2.VideoCapture camera
    :param resolution: tuple of ints of (width, height)
    :return: resolution tuple of ints (either same or default)
    """
    if not resolution:
        # pulls native res from cam and makes tuple of ints for VideoWriter object
        resolution = (int(camera_object.get(3)), int(camera_object.get(4)))
    
    camera_object.set(3, resolution[0])
    camera_object.set(4, resolution[1])
    return resolution

def record_videofeed(camera_object, filename, resolution=None):
    """
    Takes in camera object and records to filename until stopped with q command
    :param camera_object: cv2.VideoCapture of device
    :param filename: filename to record to
    :return: filename.avi in $PWD
    """
    
    cam_checker(camera_object)
    
    resolution = resolution_setter(camera_object, resolution=resolution)
    filename += '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    output = cv2.VideoWriter(filename, fourcc, 30.0, resolution) # output name, encoding, FPS, resolution tuple


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

def preview_window(camera_object, resolution=None):
    """
    Creates preview window to align webcam. Quit with Q keyboard interupt
    :param camera_object: cv2.VideoCapture to align
    :param resolution: (width, height) in ints
    """
    cam_checker(camera_object)
    resolution_setter(camera_object, resolution)
    
    while camera_object.isOpened():
        ret, frame = camera_object.read()
        if ret:
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    webcam = cv2.VideoCapture(0)
    # camera_module = cv2.VideoCapture(usePiCamera=True)
    
    resolution = (1280, 720)
    
    preview_window(webcam, resolution)
    record = input('Record? y/n')
    if record == 'y':
        record_videofeed(webcam, "day_1", resolution)