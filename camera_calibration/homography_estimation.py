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

filename='Dataset1/Pic_'
#----------------------------------------------------------------------------------------------#
#This function reads the matches saved in a text file.
def readcorners(filename):    
    f = open(filename).read()
    rows = []
    for line in f.split('\n'):
        rows.append(line.split('\t'))
    
    for loopVar1 in range(0, len(rows)):
    	for loopVar2 in range(0, len(rows[loopVar1])):
    		try:
    			rows[loopVar1][loopVar2]=float(rows[loopVar1][loopVar2]) 
    		except ValueError:
    			print rows[loopVar1][loopVar2]	  
    return rows 
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#This function saves the homographies to a text file to be read by Stitching program 
def save_homography(filename, H):
	fo = open(filename, 'w', 0)
	for loopVar1 in range(0, len(H)):
		fo.write(str(H[loopVar1][0]))
		fo.write('\t')
		fo.write(str(H[loopVar1][1]))
		fo.write('\t')
		fo.write(str(H[loopVar1][2]))
		if loopVar1 !=len(H)-1:
			fo.write('\n')
	fo.close()	
						
#---------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------#
#Points used for homography calculations - op are in original plane and tp are in transformed plane
#op=matrix([[210, 490],[371, 490],[363, 562],[219, 562]])
#tp=matrix([[116, 56],[428, 56],[428, 271],[116, 271]])

op=matrix(readcorners('world_coordinates.txt'))
for loopVar0 in range(1, 41):
	tp=matrix(readcorners(filename+str(loopVar0)+'_corners.txt'))
	print len(op), len(tp)

#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
	#Calculate the H Matrix from AH=y
	A=[] # A Matrix
	y=[] # y Matrix
	NUM_OF_POINTS=op.shape[0]# Find the size of Matrix A

	#This loop fills in Matrix A and y
	for loopVar1 in range(0,NUM_OF_POINTS):
		A.append([op[loopVar1,0],op[loopVar1,1],1,0,0,0,-op[loopVar1,0]*tp[loopVar1,0],-op[loopVar1,1]*tp[loopVar1,0]])
		A.append([0,0,0,op[loopVar1,0],op[loopVar1,1],1,-op[loopVar1,0]*tp[loopVar1,1],-op[loopVar1,1]*tp[loopVar1,1]])
		y.append([tp[loopVar1,0]])
		y.append([tp[loopVar1,1]])
	
	#Find out least squares solution and fill in H matrix
	x=linalg.lstsq(A, y)[0]
	H=matrix([[x[0][0],x[1][0],x[2][0]],[x[3][0],x[4][0],x[5][0]],[x[6][0],x[7][0],1]])
	inv_H=inv(H)
	array_H=asarray(H)
	print H
	#print array_H, array_H[0], array_H[1], array_H[0][0]
	save_homography(filename+str(loopVar0)+'_homography.txt', array_H)
#---------------------------------------------------------------------------------------------#
'''
	# Load the image in color
	#print filename+str(loopVar1)+'.jpg' 
	img_1a = cv2.imread(filename+str(loopVar0)+'_corners.jpg',1)
	#cv2.imshow('image', img_1a) #Display the result
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

	a=img_1a.shape[0]
	b=img_1a.shape[1]
	corners=[[0, 0],[0, b],[a, 0],[a, b]]
	for loopVar1 in range(0,len(corners)):
		old_point=[[corners[loopVar1][0]],[corners[loopVar1][1]],[1]]
		#print old_point
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
		#print loopVar1


	#print min_x
	#print min_y
	#print max_x
	#print max_y
	scaling_x=(max_x-min_x)/img_1a.shape[0]	# Scaling factor 
	a=img_1a.shape[0]	# Keep x coordinate same
	b=int((max_y-min_y)/scaling_x) # Scale y appropriately
	output_img=[] 
	output_img = zeros((a,b,3), uint8) # Output Image with all pixels set to black
	#print a,b
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
			
			#When indices are positive, copy the input image pixel
			if ((old_point[0][0]>0)and(old_point[1][0]>0)):
				try:
					output_img[loopVar1][loopVar2]=img_1a[old_point[0][0]][old_point[1][0]]
				#When indices exceed the available image size,keep the black pixel as it is in the output image	
				except IndexError:	
					output_img[loopVar1][loopVar2]=output_img[loopVar1][loopVar2]
			#When indices are negative, keep the black pixel as it is in the output image
			else:
					output_img[loopVar1][loopVar2]=output_img[loopVar1][loopVar2]
		print loopVar1	
#---------------------------------------------------------------------------------------------#					
	cv2.imshow('image', output_img) #Display the result
	cv2.imwrite(filename+str(loopVar0)+'_corrected.jpg', output_img) #Save the result
	cv2.waitKey(0) #Wait for key-press
	#cv2.destroyAllWindows() #Close all windows'''
