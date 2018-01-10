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
    keypointSize = 15
    kpsPerDirection = 196   # just working because all images got the same size. if they are not, dimension error occurs at filling x_train
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
#trainImages = glob.glob('./images/db/train/**/*.jpg')


descriptors = []
keypoints = create_keypoints(256, 256)
sift = cv2.xfeatures2d.SIFT_create()

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers

labels = {1: 'car', 2: 'face', 3: 'flower'}
trainImages = list()
print("Read images and assign them to their labels")
for labelNum, labelCaption in labels.items():
    imgPaths = glob.glob('./images/db/train/{}/*.jpg'.format(labelCaption + 's'))
    for imgPath in imgPaths:
        img = cv2.imread(imgPath, 1)
        trainImages.append((img, labelNum))
print("Done with image gold classification")

print("Start computing all descriptors for every image and build the training vectors")
# to get the x_train shape get descriptor of the first image
X_train = np.asmatrix(create_descriptor(trainImages[0][0]).ravel())
y_train = trainImages[0][1]
for img, imgClass in trainImages[1:]:
    X_train = np.vstack((X_train, create_descriptor(img).ravel()))
    y_train = np.vstack((y_train, imgClass))
print("Done, building training vectors")

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)
print("Now train that shizzle")
classifier = svm.LinearSVC()
classifier.fit(X_train, y_train.ravel())    # because sklearn asks for, y_train is raveld again
print("Ahhh that was a good workout. Finished the training!")
# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
# 5. output the class + corresponding name
print("Now, let's verify! Predict the test images")
testImages = glob.glob('./images/db/test/*.jpg')
for testImgPath in testImages:
    testImg = cv2.imread(testImgPath, 1)
    testDesc = create_descriptor(testImg).ravel()
    pred = classifier.predict([testDesc])
    print("The image {} is predicted as {}, which means {}.".format(testImgPath, pred[0], labels.get(pred[0])))


