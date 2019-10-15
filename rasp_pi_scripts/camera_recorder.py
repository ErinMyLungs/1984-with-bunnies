import numpy as np
import cv2

webcam = cv2.VideoCapture(0)
# camera_module = cv2.VideoCapture(usePiCamera=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID') # this is currently broken
output = cv2.VideoWriter('output_test.avi', fourcc, 30.0, (720, 1280)) # output name, encoding, FPS, resolution tuple


while(webcam.isOpened()):
    # Capture frame-by-frame
    ret, frame = webcam.read()

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