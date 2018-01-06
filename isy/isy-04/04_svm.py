import numpy as np
import cv2
import glob
from sklearn import svm


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11
    # Erhöhung der KPs bringt nicht wirklich mehr erfolg...
    kpsPerDirection = 15
    stepDistW = w / kpsPerDirection
    stepDistH = h / kpsPerDirection
    y = 0
    x = 0
    while y <= h:
        while x <= w:
            keypoints.append(cv2.KeyPoint(x, y, keypointSize))
            x += stepDistW
        x = 0
        y += stepDistH
    return keypoints

def create_descriptor(img):
    """ Creates a descriptor for a given image
    """
    keypoints = create_keypoints(img.shape[1], img.shape[0])
    _, desc = sift.compute(img, keypoints)
    return desc

# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px
trainImages = glob.glob('./images/db/train/**/*.jpg')
descriptors = []
keypoints = create_keypoints(256, 256)
sift = cv2.xfeatures2d.SIFT_create()
for imgPath in trainImages:
    img = cv2.imread(imgPath, 1)
    descriptors.append((img, create_descriptor(img)))

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
x_train = np.zeros((len(trainImages), len(keypoints) * ))

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)


# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image

# 5. output the class + corresponding name
