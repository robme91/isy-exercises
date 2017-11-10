import numpy as np
import cv2
import math
import sys
from ImageStitcher import *

############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images
carPanoR = cv2.imread('images/pano3.jpg', 1)
carPanoM = cv2.imread('images/pano2.jpg', 1)
carPanoL = cv2.imread('images/pano1.jpg', 1)

cityPanoR = cv2.imread('images/pano6.jpg', 1)
cityPanoM = cv2.imread('images/pano5.jpg', 1)
cityPanoL = cv2.imread('images/pano4.jpg', 1)

# order of input images is important(from right to left)
imageStitcherCar = ImageStitcher([carPanoR, carPanoM, carPanoL]) # list of images
(matchlistCar, resultCar) = imageStitcherCar.stitch_to_panorama()

imageStitcherCity = ImageStitcher([cityPanoR, cityPanoM, cityPanoL]) # list of images
(matchlistCity, resultCity) = imageStitcherCity.stitch_to_panorama()

if not matchlistCar and matchlistCity:
    print("We have not enough matching keypoints to create a panorama")
else:
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image
    for idx, out in enumerate(matchlistCar):
        cv2.imshow("Match car" + str(idx), out)
    cv2.imshow("Panorama Ergebnis car", resultCar)

    for idx, out in enumerate(matchlistCity):
        cv2.imshow("Match city" + str(idx), out)
    cv2.imshow("Panorama Ergebnis city", resultCity)
    while True:
        key = cv2.waitKey(0) & cv2.waitKey(0xFF)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    print('end')
