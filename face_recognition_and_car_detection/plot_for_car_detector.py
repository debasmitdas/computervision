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


x=[1,2,3,4,5,6,7,8]
fp_train=[ 0.41638225256,  0.18771331058,  0.0927189988623,  0.042662116041,  0.014220705347,  0.0028441410694, 0.00113765642776,  0.0]
fn_train=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

fp_test=[ 0.42954545,  0.20454545,  0.10227273,  0.06590909,  0.04772727,  0.04545455, 0.04545455,  0.04545455]
fn_test=[ 0.01685393,  0.02247191,  0.03932584,  0.05617978,  0.05617978,  0.06179775, 0.07865169,  0.07865169]

print len(x), len(fp_train), len(fn_train), len(fp_test), len(fn_test)

fig_train, ax_train = plt.subplots()
ax_train.plot(x, fp_train, label="FP")
ax_train.plot(x, fn_train, label="FN")
ax_train.legend(loc=1); # upper left corner
ax_train.set_xlabel('Number of Stages')
ax_train.set_ylabel('Rate')
ax_train.set_title('Trend of FP and FN rate with Number of Stages on Training Data');
plt.show()

fig_test, ax_test = plt.subplots()
ax_test.plot(x, fp_test, label="FP")
ax_test.plot(x, fn_test, label="FN")
ax_test.legend(loc=1); # upper left corner
ax_test.set_xlabel('Number of Stages')
ax_test.set_ylabel('Rate')
ax_test.set_title('Trend of FP and FN rate with Number of Stages on Testing Data');
plt.show()

