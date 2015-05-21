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
filename2='set1_img2_output_pd.jpg'
filename3='Img2.jpg'
filename4='set1_img2_final_output.jpg'
#----------------------------------------------------------------------------------------------#
#Points used for homography calculations - cp are corner points of a world-rectangle in image plane
#The points are different for different sets and images
#Img1
cp=[[94, 320],[150, 470],[455, 270],[442, 432]]
#Img2
#cp=[[192, 425],[180, 498],[331, 434],[334, 506]]
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

NUM_OF_POINTS=len(cp)
for loopVar1 in range(0,NUM_OF_POINTS):		# Convert all points to homogeneous coordinates
	cp[loopVar1]=conv_to_homogeneous(cp[loopVar1])
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#Calculate the vanishing line using four corners of a world-rectangle by cross products
#Find Hp matrix
l1=cross_product(cp[0], cp[1])
l2=cross_product(cp[2], cp[3])
l3=cross_product(cp[0], cp[2])
l4=cross_product(cp[1], cp[3])
P=cross_product(l1, l2)
Q=cross_product(l3, l4)
vl=cross_product(P, Q)
print vl

Hp=matrix([[1,0,0],[0,1,0],[vl[0],vl[1],vl[2]]])
print Hp
inv_Hp=inv(Hp)
print inv_Hp
tranpose_inv_Hp=transpose(inv_Hp)
print tranpose_inv_Hp

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
	new_point=Hp*old_point
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
		old_point=inv_Hp*new_point
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
cv2.imwrite(filename2, output_img) #Save the result'''

#---------------------------------------------------------------------------------------------#
#Stage2
new_l1=tranpose_inv_Hp*transpose(matrix(l1))
new_l2=tranpose_inv_Hp*transpose(matrix(l2))
new_m2=tranpose_inv_Hp*transpose(matrix(l3))
new_m1=tranpose_inv_Hp*transpose(matrix(l4))
new_l1=conv_2Dmat_to_arr(new_l1)
new_l2=conv_2Dmat_to_arr(new_l2)
new_m1=conv_2Dmat_to_arr(new_m1)
new_m2=conv_2Dmat_to_arr(new_m2)
print new_l1
print new_l2
print new_m1
print new_m2
#---------------------------------------------------------------------------------------------#
#Calculate the Ha Matrix from A(Ha)=y
A=[] # A Matrix
y=[] # y Matrix

# Fill in Matrix A and y
A.append([new_l1[0]*new_m1[0], (new_l1[0]*new_m1[1]+new_l1[1]*new_m1[0])])
A.append([new_l2[0]*new_m2[0], (new_l2[0]*new_m2[1]+new_l2[1]*new_m2[0])])
y.append([-new_l1[1]*new_m1[1]])
y.append([-new_l2[1]*new_m2[1]])
	
#Find out least squares solution and fill in S matrix
x=linalg.lstsq(A, y)[0]
print 'A',A
print 'x',x
print 'y',y
S=matrix([[x[0][0], x[1][0]],[x[1][0], 1]])
print 'S',S
#---------------------------------------------------------------------------------------------#
#Find SVD decomposition of S and then find A and then Ha 
U, D, V = linalg.svd(S, full_matrices=True)
D=matrix([[D[0], 0],[0, D[1]]])
print 'U',U
print 'D',D
print 'V',V
A=U*sqrt(D)*transpose(U)
print 'A',A
print 'AAT',A*transpose(A)
Ha=matrix([[A[0,0],A[0,1],0],[A[1,0],A[1,1],0],[0,0,1]])
print Ha
inv_Ha=inv(Ha)
print inv_Ha
H_total=inv_Ha*Hp
print H_total
inv_H_total=inv(H_total)
if filename1==filename3:
	H=H_total
	inv_H=inv_H_total
else:
	H=inv_Ha
	inv_H=Ha		
input_img = cv2.imread(filename3,1)
#---------------------------------------------------------------------------------------------#
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
	new_point=H*old_point
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
		old_point=inv_H*new_point
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
cv2.imwrite(filename4, output_img) #Save the result
cv2.waitKey(0) #Wait for key-press
cv2.destroyAllWindows() #Close all windows'''
