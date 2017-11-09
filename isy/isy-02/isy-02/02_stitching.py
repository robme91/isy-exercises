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

# order of input images is important(from right to left)
imageStitcher = ImageStitcher([carPanoR, carPanoM, carPanoL]) # list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image
    print('end')
