# module to connect to and record from video camera
import numpy as np
import time
import cv2
from datetime import datetime
import os

class recorder:
    def __init__(self, camera_object, resolution=None):
        """
        :param camera_object:
        :param resolution:
        """
        self.camera_object = camera_object
        self.resolution = resolution

    def _cam_checker(self):
        """
        basic check if camera_object is opened
        :param camera_object: cv2.VideoCapture device
        """
        if not self.camera_object.isOpened():
            print("Camera not opened!")
            raise ValueError
        else:
            return True

    def _resolution_setter(self):
        """
        Sets resolution from argument or gets default camera resolution
        :param camera_object: cv2.VideoCapture camera
        :param resolution: tuple of ints of (width, height)
        :return: resolution tuple of ints (either same or default)
        """
        if not self.resolution:
            # pulls native res from cam and makes tuple of ints for VideoWriter object
            resolution = (int(self.camera_object.get(3)), int(self.camera_object.get(4)))

        self.camera_object.set(3, resolution[0])
        self.camera_object.set(4, resolution[1])
        return resolution

    def record_videofeed(self, time_record=1, display_preivew=False, **kwargs):
        """
        Takes in camera object and records to filename until stopped with q command
        :param camera_object: cv2.VideoCapture of device
        :param resolution: (width:int, height:int) vid resolution to record at
        :param time_record: how long to record either float or int. 1 = 1 hour.
        :return: filename.avi in $PWD
        """

        self._cam_checker()
        resolution = self._resolution_setter()


        fourcc = cv2.VideoWriter_fourcc(*'X264') #raspberry pi encoder settings
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG') # laptop encoder settings
        output = cv2.VideoWriter('output.avi', fourcc, 30.0, resolution) # output name, encoding, FPS, resolution tuple

        time_end = time.time() + 60 * 60 * time_record #makes end time by seconds/min * min/hour * hours to rec.

        while(self.camera_object.isOpened()) and time.time() < time_end:
            # Capture frame-by-frame
            ret, frame = self.camera_object.read()

            if ret == True:
                output.write(frame)
                if display_preivew:
                    cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break

        # When everything done, release the output
        output.release()
        cv2.destroyAllWindows()

    def preview_window(self):
        """
        Creates preview window to align webcam. Quit with Q keyboard interupt
        :param camera_object: cv2.VideoCapture to align
        :param resolution: (width, height) in ints
        """
        self._cam_checker()
        self._resolution_setter()

        while self.camera_object.isOpened():
            ret, frame = self.camera_object.read()
            if ret:
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): # possibly update waitkey to 33.3?
                    break
            else:
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # TODO: Make this own function? Right now just brittle but future refactoring
    camera_number = int(input('camera hardware number: 0 if rpcam, 1 if webcam: '))

    webcam = cv2.VideoCapture(camera_number)
    # TODO: Hook up picam module? Or webcam. Or both.
    # camera_module = cv2.VideoCapture(usePiCamera=True)
    
    resolution = (1280, 720)
    
    preview_window(webcam, None)
    record = input('Record? y/n: ')
    length = input('How long to record(hours)?: ')
    display = bool(input('display preview?: '))
    if record == 'y':
        try:
            length = int(length)
        except ValueError:
            print('length has to be int coercible')

        for _ in range(length):
            current_dt = datetime.now()
            filename_string = f'{current_dt.month}-{current_dt.day}-{current_dt.year}-{current_dt.time().isoformat()[:8]}.avi'

            record_videofeed(webcam, resolution, time_record=1, display_preivew=display)
            os.rename('output.avi', filename_string)
    webcam.release()
