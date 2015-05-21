# Importing libraries
import cv2
from math import *
from numpy import *
from sympy import Symbol,cos,sin

from operator import *
from numpy.linalg import *
import time
import ctypes

# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

#----------------------------------------------------------------------------------------------#
#Filenames. These names change for different sets and images
filename1='Img2.jpg'
filename2='set1_img2_direct_output.jpg'
#----------------------------------------------------------------------------------------------#
#Points used for homography calculations - cp are corner points of a world-rectangle in image plane
#The points are different for different sets and images
#Img1
'''cp1=[[93, 319],[149, 469],[441, 430]]
cp2=[[93, 319],[454, 270],[441, 430]]
cp3=[[454, 270],[93, 319],[149, 469]]
cp4=[[454, 270],[441, 430],[149, 469]]
cp5=[[256, 700],[376, 684],[377, 720]]#'''
#Img2
#cp=[[192, 425],[180, 498],[331, 434],[334, 506]]
cp1=[[192, 425],[180, 498],[334, 506]]
cp2=[[192, 425],[331, 434],[334, 506]]
cp3=[[331, 434],[192, 425],[180, 498]]
cp4=[[331, 434],[334, 506],[180, 498]]
cp5=[[180, 374],[392, 387],[382, 318]]#'''
#----------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#Some frequently used operations as functions
def conv_to_homogeneous(a):
	homogeneous_a=[a[0],a[1],1]
	return homogeneous_a

def conv_2Dmat_to_arr(a):
	array=[a[0,0],a[1,0],a[2,0]]	
	return array
	
def cross_product(a,b):
	result=[a[1]*b[2]-a[2]*b[1], -(a[0]*b[2]-a[2]*b[0]), a[0]*b[1]-a[1]*b[0]]
	if result[2]!=0:
		result=[result[0]/float(result[2]), result[1]/float(result[2]), result[2]/float(result[2])]
	return result
#---------------------------------------------------------------------------------------------#

NUM_OF_POINTS=len(cp1)
for loopVar1 in range(0,NUM_OF_POINTS):		# Convert all points to homogeneous coordinates
	cp1[loopVar1]=conv_to_homogeneous(cp1[loopVar1])
	cp2[loopVar1]=conv_to_homogeneous(cp2[loopVar1])
	cp3[loopVar1]=conv_to_homogeneous(cp3[loopVar1])
	cp4[loopVar1]=conv_to_homogeneous(cp4[loopVar1])
	cp5[loopVar1]=conv_to_homogeneous(cp5[loopVar1])
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#Calculate the vanishing line using four corners of a world-rectangle by cross products
#Find Hp matrix
l1=cross_product(cp1[0], cp1[1])
m1=cross_product(cp1[1], cp1[2])

l2=cross_product(cp2[0], cp2[1])
m2=cross_product(cp2[1], cp2[2])

l3=cross_product(cp3[0], cp3[1])
m3=cross_product(cp3[1], cp3[2])

l4=cross_product(cp4[0], cp4[1])
m4=cross_product(cp4[1], cp4[2])

l5=cross_product(cp5[0], cp5[1])
m5=cross_product(cp5[1], cp5[2])

#---------------------------------------------------------------------------------------------#
#Calculate the C_inf Matrix
A=[] # A Matrix
y=[] # y Matrix

# Fill in Matrix A and y
A.append([l1[0]*m1[0], (l1[0]*m1[1]+l1[1]*m1[0])/2, l1[1]*m1[1], (l1[0]*m1[2]+l1[2]*m1[0])/2, (l1[1]*m1[2]+l1[2]*m1[1])/2])
A.append([l2[0]*m2[0], (l2[0]*m2[1]+l2[1]*m2[0])/2, l2[1]*m2[1], (l2[0]*m2[2]+l2[2]*m2[0])/2, (l2[1]*m2[2]+l2[2]*m2[1])/2])
A.append([l3[0]*m3[0], (l3[0]*m3[1]+l3[1]*m3[0])/2, l3[1]*m3[1], (l3[0]*m3[2]+l3[2]*m3[0])/2, (l3[1]*m3[2]+l3[2]*m3[1])/2])
A.append([l4[0]*m4[0], (l4[0]*m4[1]+l4[1]*m4[0])/2, l4[1]*m4[1], (l4[0]*m4[2]+l4[2]*m4[0])/2, (l4[1]*m4[2]+l4[2]*m4[1])/2])
A.append([l5[0]*m5[0], (l5[0]*m5[1]+l5[1]*m5[0])/2, l5[1]*m5[1], (l5[0]*m5[2]+l5[2]*m5[0])/2, (l5[1]*m5[2]+l5[2]*m5[1])/2])
y.append([-l1[2]*m1[2]])
y.append([-l2[2]*m2[2]])
y.append([-l3[2]*m3[2]])
y.append([-l4[2]*m4[2]])
y.append([-l5[2]*m5[2]])		
#Find out least squares solution and fill in S matrix
x=linalg.lstsq(A, y)[0]
print 'A',A
print 'x',x
print 'y',y
C=matrix([[x[0][0], x[1][0]/2, x[3][0]/2],[x[1][0]/2, x[2][0], x[4][0]/2],[x[3][0]/2, x[4][0]/2, 1]])
print 'C',C
S=matrix([[x[0][0], x[1][0]/2],[x[1][0]/2, x[2][0]]])
print 'S',S
if linalg.det(S)<0:
	print 'not pd'
print linalg.det(S)
#---------------------------------------------------------------------------------------------#
#Find SVD decomposition of S and then find A and then H
U, D, V = linalg.svd(S, full_matrices=True)
D=matrix([[D[0], 0],[0, D[1]]])
print 'U',U
print 'D',D
print 'V',V
A=U*sqrt(D)*transpose(U)
print 'A',A
print 'AAT',A*transpose(A)
v=inv(A)*transpose(matrix([x[3][0]/2,x[4][0]/2]))
H=matrix([[A[0,0],A[0,1],0],[A[1,0],A[1,1],0],[v[0,0],v[1,0],1]])
print H
inv_H=inv(H)
print inv_H	
#---------------------------------------------------------------------------------------------#

# Load the input image
input_img = cv2.imread(filename1,1)

#Declare two empty arrays for storing points while applying homography
old_point=[]
new_point=[]

#---------------------------------------------------------------------------------------------#
#This section of code finds out minimum and maximum indices in both x and y. 
#Later, finds out the scaling factor to scale the final output image
#Creates an empty output image with scaled-down dimensions
min_x=0
min_y=0
max_x=0
max_y=0

a=input_img.shape[0]
b=input_img.shape[1]
corners=[[0, 0],[0, b],[a, 0],[a, b]]
for loopVar1 in range(0,len(corners)):
	old_point=[[corners[loopVar1][0]],[corners[loopVar1][1]],[1]]
	print old_point
	new_point=inv_H*old_point
	new_point=new_point*(1/new_point[2][0])
	old_point=array(old_point)
	new_point=array(new_point)
	if (loopVar1==0):
		min_x=new_point[0][0]
		min_y=new_point[1][0]
		max_x=new_point[0][0]
		max_y=new_point[1][0]	
	else:
		if(new_point[0][0]<min_x):
			min_x=new_point[0][0]
		if(new_point[1][0]<min_y):
			min_y=new_point[1][0]
		if(new_point[0][0]>max_x):
			max_x=new_point[0][0]
		if(new_point[1][0]>max_y):
			max_y=new_point[1][0]
	print loopVar1

print min_x
print min_y
print max_x
print max_y
scaling_x=(max_x-min_x)/input_img.shape[0]	# Scaling factor 
a=input_img.shape[0]	# Keep x coordinate same
b=int((max_y-min_y)/scaling_x) # Scale y appropriately
output_img=[] 
output_img = zeros((a,b,3), uint8) # Output Image with all pixels set to black
print a,b
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#This loop applies inverse homography to all points in output image and gets original pixel coordinates.
#Used Inverse transformation to avoid having empty pixels in the output image
for loopVar1 in range(0,a):
	for loopVar2 in range(0,b):
		new_point=matrix([[(loopVar1*scaling_x)+min_x],[(loopVar2*scaling_x)+min_y],[1]])
		old_point=H*new_point
		old_point=old_point*(1/old_point[2][0])
		old_point=array(old_point)
		new_point=array(new_point)
		
		#When indices are positive, copy the Frame/Wall.
		if ((old_point[0][0]>0)and(old_point[1][0]>0)): 
			try:
				output_img[loopVar1][loopVar2]=input_img[old_point[0][0]][old_point[1][0]]	
			#When indices exceed the available image size,keep the black pixel as it is in the output image.	
			except IndexError:
				output_img[loopVar1][loopVar2]=output_img[loopVar1][loopVar2]
		#When indices are negative, keep the black pixel as it is in the output image.		
		else:
			output_img[loopVar1][loopVar2]=output_img[loopVar1][loopVar2]
	print loopVar1		
#---------------------------------------------------------------------------------------------#
				
cv2.imshow('image', output_img) #Display the result
cv2.imwrite(filename2, output_img) #Save the result
cv2.waitKey(0) #Wait for key-press
cv2.destroyAllWindows() #Close all windows'''
