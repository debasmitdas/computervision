# Importing libraries
import cv2
import cv
from math import *
from numpy import *
#from sympy import Symbol,cos,sin
from operator import *
from numpy.linalg import *
import time
import ctypes
from scipy.optimize import leastsq
from matplotlib import pyplot as plt
# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)


x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y_PCA=[0.287301587302, 0.78253968254, 0.946031746032, 0.973015873016, 0.988888888889, 0.987301587302, 0.990476190476, 0.990476190476, 0.992063492063, 0.996825396825, 0.996825396825, 0.996825396825, 0.998412698413, 0.998412698413, 0.998412698413]
y_LDA=[0.0936507936508, 0.611111111111, 0.92380952381, 0.97619047619, 0.990476190476, 1.0, 0.998412698413, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
print len(x), len(y_PCA), len(y_LDA)

fig, ax = plt.subplots()
ax.plot(x, y_PCA, label="PCA")
ax.plot(x, y_LDA, label="LDA")
ax.legend(loc=1); # upper left corner
ax.set_xlabel('Sub-space Dimensions')
ax.set_ylabel('Accuracy')
ax.set_title('PCA vs LDA Comparision on Test Images');
plt.show()


