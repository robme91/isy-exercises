import cv2
import glob
import numpy as np
from queue import PriorityQueue

############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
def distance(a, b):

    sum = 0.0
    for aVec, bVec in zip(a,b):
        for aX, pY in zip(aVec, bVec):
            sum += (aX - pY) ** 2
    return np.sqrt(sum)


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11
    # Erhöhung der KPs bringt nicht wirklich mehr erfolg...
    kpsPerDirection = 25
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


def create_prio_queue(img, descriptors):
    """
    creates priority queue on given image,
    and returns the queue and the original image
    """
    q = PriorityQueue()
    imgDesc = create_descriptor(img)
    for i, d in descriptors:
        dist = distance(imgDesc, d)
        q.put((dist, i))
    return q

# 1. preprocessing and load
images = glob.glob('./images/db/*/*.jpg')

# 2. create keypoints on a regular grid (cv2.KeyPoint(c, r, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
sift = cv2.xfeatures2d.SIFT_create()
for imgPath in images:
    img = cv2.imread(imgPath, 1)
    descriptors.append((img, create_descriptor(img)))

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())


# 5. output (save and/or display) the query results in the order of smallest distance
while True:
    prio = 1
    q = PriorityQueue()
    cv2.destroyAllWindows()
    car = cv2.imread('./images/db/query_car.jpg', 1)
    cv2.imshow("Original Car", car)
    face = cv2.imread('./images/db/query_face.jpg', 1)
    cv2.imshow("Original Face", face)
    flower = cv2.imread('./images/db/query_flower.jpg', 1)
    cv2.imshow("Original Flower", flower)
    print("Press 'c' for car search, 'f' for face search, 'b' for flower search or 'q' to quit")
    key = cv2.waitKey(0) & cv2.waitKey(0xFF)
    if key == ord('q'):
        break
    elif key == ord('c'):
        q = create_prio_queue(car, descriptors)
    elif key == ord('f'):
        q = create_prio_queue(face, descriptors)
    elif key == ord('b'):
        q = create_prio_queue(flower, descriptors)
    while not q.empty():
        dist, img = q.get()
        title = "Prio " + str(prio)
        cv2.imshow(title, img)
        prio += 1
        key = cv2.waitKey(0) & cv2.waitKey(0xFF)
        if key == ord('q'):
            break
cv2.destroyAllWindows()