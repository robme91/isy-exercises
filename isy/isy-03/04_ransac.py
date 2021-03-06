import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RansacPointGenerator:
    """generates a set points - linear distributed + a set of outliers"""
    def __init__(self, numpointsInlier, numpointsOutlier):
        self.numpointsInlier = numpointsInlier
        self.numpointsOutlier = numpointsOutlier
        self.points = []

        pure_x = np.linspace(0, 1, numpointsInlier)
        pure_y = np.linspace(0, 1, numpointsInlier)
        noise_x = np.random.normal(0, 0.025, numpointsInlier)
        noise_y = np.random.normal(0, 0.025, numpointsInlier)

        outlier_x = np.random.random_sample((numpointsOutlier,))
        outlier_y = np.random.random_sample((numpointsOutlier,))

        points_x = pure_x + noise_x
        points_y = pure_y + noise_y
        points_x = np.append(points_x, outlier_x)
        points_y = np.append(points_y, outlier_y)

        self.points = np.array([points_x, points_y])

class Line:
    """helper class"""
    def __init__(self, a, b):
        # y = mx + b
        self.m = a
        self.b = b


class Ransac:
    """RANSAC class. """
    def __init__(self, points, threshold):
        self.points = points
        self.threshold = threshold
        self.best_model = Line(1, 0)
        self.best_inliers = []
        self.best_score   = 1000000000
        self.current_inliers = []
        self.current_model   = Line(1, 0)
        self.num_iterations  = int(self.estimate_num_iterations(0.99, 0.5, 2))
        self.iteration_counter = 0


    def estimate_num_iterations(self, ransacProbability, outlierRatio, sampleSize):
        """
        Helper function to generate a number of generations that depends on the probability
        to pick a certain set of inliers vs. outliers.
        See https://de.wikipedia.org/wiki/RANSAC-Algorithmus for more information

        :param ransacProbability: std value would be 0.99 [0..1]
        :param outlierRatio: how many outliers are allowed, 0.3-0.5 [0..1]
        :param sampleSize: 2 points for a line
        :return:
        """
        return math.ceil(math.log(1-ransacProbability) / math.log(1-math.pow(1-outlierRatio, sampleSize)))

    def estimate_error(self, p, line):
        """
        Compute the distance of a point p to a line y=mx+b
        :param p: Point
        :param line: Line y=mx+b
        :return:
        """
        return math.fabs(line.m * p[0] - p[1] + line.b) / math.sqrt(1 + line.m * line.m)


    def step(self, iter):
        """
        Run the ith step in the algorithm. Collects self.currentInlier for each step.
        Sets if score < self.bestScore
        self.bestModel = line
        self.bestInliers = self.currentInlier
        self.bestScore = score

        :param iter: i-th number of iteration
        :return:
        """
        self.current_inliers = []

        # sample two random points from point set
        #TODO shall those two points be also included to the current inliers?
        rand1 = np.random.randint(0, len(self.points[0]) - 1)
        rand2 = np.random.randint(0, len(self.points[0]) - 1)
        point1 = self.points[0:, rand1]
        point2 = self.points[0:, rand2]
        # find two definitely different points
        while np.array_equal(point1, point2):
            rand2 = np.random.randint(0, len(self.points[0]) - 1)
            point2 = self.points[0:, rand2]

        # compute line parameters m / b and create new line
        point1_x, point1_y = point1
        point2_x, point2_y = point2
        m = (point2_y - point1_y) / (point2_x - point1_x)
        b = m * (- point1_x) + point1_y
        line = Line(m, b)

        # loop over all points
        # compute error of all points and add to inliers if
        # err smaller than threshold update score, otherwise add error/threshold to score
        #TODO don't unerstand what shall be added to score here. Must error always sumed to score or is the "error/threshold" a math operation? Then why i should do this
        score = 0
        for idx in range(0, len(self.points[0])):
            point = self.points[0:, idx]
            err = self.estimate_error(point, line)
            if err < self.threshold:
                self.current_inliers.append(point)
            score += err

        # if score < self.bestScore: update the best model/inliers/score
        # please do look at resources in the internet :)
        # TODO why? in script it is the size of inliers that should be used
        if len(self.current_inliers) > len(self.best_inliers):
            self.best_inliers = self.current_inliers
            self.best_model = line
            self.best_score = score

        print(iter, "  :::::::::: bestscore: ", self.best_score, " bestModel: ", self.best_model.m, self.best_model.b)

    def run(self):
        """
        run RANSAC for a number of iterations
        :return:
        """
        for i in range(0, self.num_iterations):
            self.step(i)


#create different point arrays
rpg1 = RansacPointGenerator(100, 45)
#print(rpg1.points)
rpg2 = RansacPointGenerator(40, 150)
#print(rpg2.points)
rpg3 = RansacPointGenerator(60, 60)
#print(rpg3.points)
rpgArr = [rpg1, rpg2, rpg3]

#create different threasholds
th1 = 0.01
th2 = 0.05
th3 = 0.5
thArr = [th1, th2, th3]

for rpg in rpgArr:
    for th in thArr:
        ransac = Ransac(rpg.points, th)
        ransac.run()

        # print rpg.points.shape[1]
        #plt.subplot().set_title('Threshold: {0} Inliers: {1} Outliers: {2}'.format(th, rpg.numpointsInlier, rpg.numpointsOutlier))
        plt.figure()
        plt.plot(rpg.points[0, :], rpg.points[1, :], 'ro')
        m = ransac.best_model.m
        b = ransac.best_model.b
        plt.plot([0, 1], [m*0 + b, m*1+b], color='k', linestyle='-', linewidth=2)
        # #
        plt.axis([0, 1, 0, 1])
        plt.title('Threshold: {0} Inliers: {1} Outliers: {2}'.format(th, rpg.numpointsInlier, rpg.numpointsOutlier))

plt.show()
