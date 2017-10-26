import numpy as np
import cv2
import math
import time


######################################################################
# IMPORTANT: Please make yourself comfortable with numpy and python:
# e.g. https://www.stavros.io/tutorials/python/
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# Note: data types are important for numpy and opencv
# most of the time we'll use np.float32 as arrays
# e.g. np.float32([0.1,0.1]) equal np.array([1, 2, 3], dtype='f')


######################################################################
# A2. OpenCV and Transformation and Computer Vision Basic

# (1) read in the image Lenna.png using opencv in gray scale and in color
# and display it NEXT to each other (see result image)
# Note here: the final image displayed must have 3 color channels
#            So you need to copy the gray image values in the color channels
#            of a new image. You can get the size (shape) of an image with rows, cols = img.shape[:2]

# why Lenna? https://de.wikipedia.org/wiki/Lena_(Testbild)

# (2) Now shift both images by half (translation in x) it rotate the colored image by 30 degrees using OpenCV transformation functions
# + do one of the operations on keypress (t - translate, r - rotate, 'q' - quit using cv::warpAffine
# http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
# Tip: you need to define a transformation Matrix M
# see result image

# Aufgabe 2.1 + 2.2
lennaCol = cv2.imread('images/Lenna.png', 1)
lennaGrey = cv2.cvtColor(cv2.imread('images/Lenna.png', 0), cv2.COLOR_GRAY2RGB)
rows, cols = lennaCol.shape[:2]
img = np.zeros((rows, cols * 2, 3), np.uint8)
img[:, 0:cols] = lennaCol
img[:, cols: np.size(img, 1)] = lennaGrey
cv2.namedWindow("Lenna")
cv2.imshow("Lenna", img)
key = cv2.waitKey(0) & cv2.waitKey(0xFF)
if key == ord('q'):
    cv2.destroyAllWindows()
elif key == ord('t'):
    print("Verschieben")
elif key == ord('r'):
    print("Rotieren")
# else:
#     print("Drücken Sie die Tasten q - zum Beenden, r - zum Rotieren oder t - zum Verschieben des Bildes")