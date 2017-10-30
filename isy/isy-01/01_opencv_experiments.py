import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    orig = frame;
    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1
    elif ch == ord('2'):
        mode = 2
    elif ch == ord('3'):
        mode = 3
    elif ch == ord('4'):
        mode = 4
    elif ch == ord('5'):
        mode = 5
    elif ch == ord('6'):
        mode = 6
    elif ch == ord('7'):
        mode = 7
    if ch == ord('q'):
        break

    if mode == 1:
        # just example code
        # your code should implement
        frame = orig
    elif mode == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif mode == 3:
        # change to LAB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    elif mode == 4:
        # change to YUV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    elif mode == 5:
        # gaussian thresholding
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.adaptiveThreshold(grayFrame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    elif mode == 6:
        # otsu thresh.
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayFrame, (5, 5), 0)
        crab, frame = cv2.threshold(grayFrame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif mode == 7:
        # canny edges
        frame = cv2.Canny(frame, 120, 180)
    # Display the resulting frame
    cv2.imshow('frame', frame)






# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()