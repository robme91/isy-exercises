import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow('Interactive Systems: Towards AR Tracking')
while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key

    # YOUR CODE HERE
    #end if q pressed

    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(grayFrame, None)
    cv2.drawKeypoints(grayFrame, kp, frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Interactive Systems: Towards AR Tracking', frame)


cap.release()
cv2.destroyAllWindows()