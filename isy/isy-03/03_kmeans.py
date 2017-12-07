import numpy as np
import cv2
import math
import sys
from collections import defaultdict


############################################################
#
#                       KMEANS
#
############################################################


##################### Whole Code inspired by Tom Oberhauser ############################################


# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    dis = 0.0
    for aX, pY in zip(a, b):
        dis += (aX - pY) ** 2
    return np.sqrt(dis)

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask, current_cluster_centers):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    clusterColors = defaultdict(list)
    img_h, img_w = img.shape[:2]
    for h in range(0, img_h):
        for w in range(0, img_w):
            clusterID = int(clustermask[h][w])
            color_pixel = img[h][w]
            clusterColors[clusterID].append(color_pixel)
    for k in clusterColors.keys():
        current_cluster_centers[k] = np.uint8(np.mean(clusterColors[k], axis=0))

def assign_to_current_mean(img, result, clustermask, current_cluster_centers):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0
    img_h, img_w = img.shape[:2]
    for h in range(0, img_h):
        for w in range(0, img_w):
            color_pixel = img[h][w]
            color_mean = current_cluster_centers[clustermask[h][w]]
            color_mean = np.reshape(color_mean, color_pixel.shape)
            overall_dist += distance(color_pixel, color_mean)
            result[h][w] = color_mean
    return overall_dist



def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    h, w = img.shape[:2]
    rand_h = int(h * np.random.random())
    rand_w = int(h * np.random.random())
    return img[rand_h][rand_w] # return color at random pixel position


def kmeans(img, current_cluster_centers):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    h1, w1 = img.shape[:2]
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max

    clustermask = np.zeros((h1, w1, 1), np.uint8)
    result = np.zeros((h1, w1, 3), np.uint8)

    # initializes each pixel to a cluster
    # iterate for a given number of iterations or if rate of change is
    # very small

    i = 0
    change_rate = 1.0
    while i < max_iter and change_rate > max_change_rate:
        changes = 0
        for h in range(0,h1):
            for w in range(0, w1):
                # get pixel and calculate distances to all current cluster centers
                pixel = img[h][w]
                currentClusterID = clustermask[h][w]
                distances = []
                # I don´t use PriorityQueue here since it crashes when to values have the same priority
                # (see last assignment, gave an additional index to the queue to have a second index
                # for it to order by).
                for id, cc in enumerate(current_cluster_centers):
                    cc = np.reshape(cc, pixel.shape)
                    distances.append((distance(cc, pixel), id))

                closestClusterID = sorted(distances, key=lambda x: x[0])[0][1] # get closest clusterID
                if closestClusterID != currentClusterID:
                    # Set id of closest cluster center to clustermask at this position
                    clustermask[h][w] = closestClusterID
                    changes+=1

        change_rate = changes / (w1*h1)
        print('Change rate: ', change_rate)
        overallError = assign_to_current_mean(img, result, clustermask, current_cluster_centers)
        #update_mean(img, clustermask, current_cluster_centers)
        print('Error: ', overallError)
        i+=1

    return result

# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

# load image
imgraw = cv2.imread('./images/Lenna.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# compare different color spaces and their result for clustering
# YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
image = imgraw
# image = cv2.cvtColor(imgraw, cv2.COLOR_BGR2YUV)
# image = cv2.cvtColor(imgraw, cv2.COLOR_BGR2LAB)
image = cv2.cvtColor(imgraw, cv2.COLOR_BGR2HSV)

for i in range(0, numclusters):
    current_cluster_centers[i] = initialize(image)

h1, w1 = image.shape[:2]

# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
# res = kmeans(image)
res = kmeans(image, current_cluster_centers)

h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()


#TODO check https://mubaris.com/2017-10-01/kmeans-clustering-in-python