import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints):

 # convert color to gray image and extract feature in gray

    imgGray = cv2.cvtColor(imgcolor, cv2.COLOR_BGRA2GRAY)
    # compute x and y gradients
    # taken from here: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    sobelX = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1)
    # compute magnitude and angle of the gradients
    mog = cv2.magnitude(sobelX, sobelY)

    # test images are kind of broken, because the blue color isn't the same in all pixels

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    for count, kp in enumerate(keypoints):
        kpY, kpX = kp.pt
        y1 = int(kpY - int(kp.size / 2))
        y2 = int(kpY + int(kp.size / 2)) + 1    # add one to get the right length when slicing later
        x1 = int(kpX - int(kp.size / 2))
        x2 = int(kpX + int(kp.size / 2)) + 1    # add one to get the right length when slicing later
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow
        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        # (hist, bins) = np.histogram(...)
        """ Answer: When there is no gradient, angle returns also 0 and not None or s.l.t which has the same meaning as
            an angle of 0. So we need to filte out all 0=Nones, that only the really 0 angles are left.
        """

        mog_window = mog[y1:y2, x1:x2]
        angle = cv2.phase(sobelX[y1:y2, x1:x2], sobelY[y1:y2, x1:x2])
        # filter out all nones, with use of np magic
        angle = angle[mog_window != 0]
        # bins is the max. of balken which are shown, range set all histos to the same width in x direction,
        #  density normalizes the values in y direction, analog to imgs in exercise
        (hist, bins) = np.histogram(angle, bins=8, range=(0.0, 2 * np.pi), density=True)
        plot_histogram(hist, bins)
        descr[count] = hist

        return descr


#increased keypoint because of the keypoint is to small for outgoing the circle
keypoints = [cv2.KeyPoint(15, 15, 25)]

# test for all test images
testImages = []
testImages.append(cv2.imread('./images/hog_test/diag.jpg'))
testImages.append(cv2.imread('./images/hog_test/horiz.jpg'))
testImages.append(cv2.imread('./images/hog_test/vert.jpg'))
testImages.append(cv2.imread('./images/hog_test/circle.jpg'))
for test in testImages:
    descriptor = compute_simple_hog(test, keypoints)

