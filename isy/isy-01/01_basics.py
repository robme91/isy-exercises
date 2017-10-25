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

#
# a) (*) Erzeugen Sie einen Vektor mit Nullen der Länge 10 (10 Elemente) und setzen den Wert des
# 5.Elementes auf eine 1.

vec = np.zeros(10)
vec[4] = 1
print("a) ", vec)
print()

# b) (*) Erzeugen Sie einen Vektor mit Ganzahl-Werten von 10 bis 49 (geht in einer Zeile).
vec = np.arange(10, 50)
print("b) ",  vec)
print()

# c) (*) Drehen Sie die Werte des Vektors um (geht in einer Zeile).
print("c) ", vec[::-1])
print()

# d) (*) Erzeugen Sie eine 4x4 Matrix mit den Werte 0 bis 15 (links oben rechts unten).
vec = np.arange(16).reshape(4,4)
print("d) ", vec)
print()

# e) (*) Erzeuge eine 8x8 Matrix mit Zufallswerte und finde deren Maximum und Minimum und
# normalisieren Sie die Werte (sodass alle Werte zwischen 0 und 1 liegen - ein Wert wird 1 (max)
# sein und einer 0 (min)).
vec = np.random.random((8, 8))
vMin, vMax = vec.max(), vec.min()
vec = (vec - vMin) / (vMax - vMin)
print("e) ")
print(vec) #warum minus null
print()

# f) (*) Multiplizieren Sie eine 4x3 Matrix mit einer 3x2 Matrix
print("f) ")
a = np.arange(1,13).reshape(4,3)
print(a)
b = np.arange(1,7).reshape(3, 2)
print(b)
ab = np.dot(a, b)
print(ab)
print()

# g) (*) Erzeugen Sie ein 1D Array mit den Werte von 0 bis 20 und negieren Sie Werte zwischen 8
# und 16 nachträglich.
print("g) ")
vec = np.arange(0, 21)
for i in vec[8:17]:
    vec[i] *= -1
print(vec)

# h) (*) Summieren Sie alle Werte in einem Array.
print("h) ")
vec = np.arange(1, 4)
print(vec.sum())

# i) (** ) Erzeugen Sie eine 5x5 Matrix und geben Sie jeweils die geraden und die ungeraden Zeile
# aus.
print("i) ")
mat = np.floor(10 * np.random.random((5, 5)))
counter = 0
print(mat)
for row in mat:
    if counter % 2 == 0:
        print("Gerade Zeile: ", row)
    else:
        print("Ungerade Zeile: ", row)
    counter += 1

# j) (** ) Erzeugen Sie eine Matrix M der Größe 4x3 und einen Vektor v mit Länge 3. Multiplizieren
# Sie jeden Spalteneintrag aus v mit der kompletten Spalte aus M. Schauen Sie sich dafür an, was
# Broadcasting in Numpy bedeutet.


# k) (** ) Erzeugen Sie einen Zufallsmatrix der Größe 10x2, die Sie als Kartesische Koordinaten
# interpretieren können ([[x0, y0],[x1, y1],[x2, y2]]). Konvertieren Sie diese in Polarkoordinaten
# https://de.wikipedia.org/wiki/Polarkoordinaten.


# l) (***) Implementieren Sie zwei Funktionen, die das Skalarprodukt und die Vektorlänge für Vektoren
# beliebiger Länge berechnen. Nutzen Sie dabei NICHT die gegebenen Funktionen von
# NumPy. Testen Sie Ihre Funktionen mit den gegebenen Vektoren:
print("l) ")
def skalar(vec1=[], vec2=[]):
    if len(vec1) == len(vec2):
        result = 0
        for idx, val in enumerate(vec1):
            result = result + val * vec2[idx]
    else:
        result = "Vektoren unterschiedlicher Länge können nicht zum Skalar gebildet werden."
    return result

print("Eigene Funktion: ", skalar([1, 2, 3, 4, 5], [-1, 9, 5, 3, 1]))
vec1 = np.array([1, 2, 3, 4, 5])
vec2 = np.array([-1, 9, 5, 3, 1])
print("Prüfung mit numpy: ", np.dot(vec1, vec2))


