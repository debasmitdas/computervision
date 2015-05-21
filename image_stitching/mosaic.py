# Importing libraries
import cv2
import cv
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
#Filenames
img_filenames=['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', 'img7.jpg',]
homography_filenames=['homography12.txt', 'homography23.txt', 'homography34.txt', 'homography45.txt', 'homography56.txt', 'homography67.txt',]
out_filename='stitched.jpg'

			
#--------------------------------------------------------------------------------------------#
#This function reads the homographies saved in a text file.
def readhomography(filename):    
    f = open(filename).read()
    rows = []
    for line in f.split('\n'):
        rows.append(line.split('\t'))
    
    for loopVar1 in range(0, len(rows)):
    	for loopVar2 in range(0, len(rows[loopVar1])):
    		rows[loopVar1][loopVar2]=float(rows[loopVar1][loopVar2])
    #f.close()		   
    return rows 
#--------------------------------------------------------------------------------------------# 

# Load computed homographies from RANSAC
H12=matrix(readhomography(homography_filenames[0]))
H23=matrix(readhomography(homography_filenames[1]))
H34=matrix(readhomography(homography_filenames[2]))
H45=matrix(readhomography(homography_filenames[3]))
H56=matrix(readhomography(homography_filenames[4]))
H67=matrix(readhomography(homography_filenames[5]))


# Find homographies between center image and peripheral images
H14=(H12*H23*H34)
H14=H14/array(H14)[2][2]
H24=H23*H34
H24=H24/array(H24)[2][2]
H44=matrix(identity(3))
H54=inv(H45)
H54=H54/array(H54)[2][2]
H64=inv(H56)*inv(H45)
H64=H64/array(H64)[2][2]
H74=inv(H67)*inv(H56)*inv(H45)
H74=H74/array(H74)[2][2]

print H14
print H24
print H34
print H44
print H54
print H64
print H74

# Load all images
images=[]
for loopVar1 in range(0, len(img_filenames)):
	images.append(cv2.imread(img_filenames[loopVar1],1))

H=[H14, H24, H34, H44, H54, H64, H74]	# Save all homographies as a list

#---------------------------------------------------------------------------------------------#
#This section of code finds out borders of each image in final canvas 
#Creates an empty output image with the dimensions of the final canvas
x_range=[]
y_range=[]
for loopVar1 in range(0, len(images)):
	
	min_x=1e+10
	min_y=1e+10
	max_x=-1e+10
	max_y=-1e+10

	a=images[loopVar1].shape[1]
	b=images[loopVar1].shape[0]
	corners=[[0, 0],[0, b],[a, 0],[a, b]]
	for loopVar2 in range(0,len(corners)):
		old_point=[[corners[loopVar2][0]],[corners[loopVar2][1]],[1]]
		print old_point
		new_point=H[loopVar1]*old_point
		new_point=new_point*(1/new_point[2][0])
		old_point=array(old_point)
		new_point=array(new_point)
		if(new_point[0][0]<min_x):
			min_x=new_point[0][0]
		if(new_point[1][0]<min_y):
			min_y=new_point[1][0]
		if(new_point[0][0]>max_x):
			max_x=new_point[0][0]
		if(new_point[1][0]>max_y):
			max_y=new_point[1][0]
	
	x_range.append([min_x,max_x])
	y_range.append([min_y,max_y])
	print loopVar2


x_range=array(x_range).astype(int)
y_range=array(y_range).astype(int)
min_x=min([item for sublist in x_range for item in sublist])
max_x=max([item for sublist in x_range for item in sublist])
min_y=min([item for sublist in y_range for item in sublist])
max_y=max([item for sublist in y_range for item in sublist])
x_range=(array(x_range)-min_x)
y_range=(array(y_range)-min_y)
print min_x, max_x, min_y, max_y
print x_range
print y_range

# Create an empty image with appropriate width and height
output_img=[] 
width=max([item for sublist in x_range for item in sublist])
height=max([item for sublist in y_range for item in sublist])
output_img = zeros((height,width,3), uint8) # Output Image with all pixels set to black
print width,height
overlapping_pixels=500
start_offset=100
#---------------------------------------------------------------------------------------------#
# This loop fills in the final image pixel by pixel by copying pixels values from corresponding parent image
for loopVar0 in range(0, len(images)):
	if loopVar0<(len(images)/2):
		x_begin=x_range[loopVar0][0]
		if loopVar0!=0:
			x_begin+=start_offset
		x_end=x_range[loopVar0+1][0]+overlapping_pixels
	elif loopVar0==(len(images)/2):
		x_begin=x_range[loopVar0][0]+start_offset
		x_end=x_range[loopVar0][1]
	else:
		x_begin=x_range[loopVar0-1][1]-overlapping_pixels
		x_end=x_range[loopVar0][1]	
	y_begin=y_range[loopVar0][0]
	y_end=y_range[loopVar0][1]
	
	print x_begin, x_end
	print y_begin, y_end
	
	inv_h=inv(H[loopVar0])
	#---------------------------------------------------------------------------------------------#
	#This loop applies inverse homography to all points in output image and gets original pixel coordinates.
	#Used Inverse transformation to avoid having empty pixels in the output image
	for loopVar1 in range(x_begin,x_end):
		for loopVar2 in range(y_begin,y_end):
			new_point=matrix([[loopVar1+min_x],[loopVar2+min_y],[1]])
			old_point=inv_h*new_point
			old_point=old_point*(1/old_point[2][0])
			old_point=array(old_point)
			new_point=array(new_point)
		
			#When indices are positive, copy the pixel.
			if ((old_point[0][0]>0)and(old_point[1][0]>0)): 
				try:
					output_img[loopVar2][loopVar1]=images[loopVar0][old_point[1][0]][old_point[0][0]]	
				#When indices exceed the available image size,keep the black pixel as it is in the output image.	
				except IndexError:
					pass
			#When indices are negative, keep the black pixel as it is in the output image.		
			else:
				pass
		print loopVar1		
#---------------------------------------------------------------------------------------------#
cv2.imshow('stitched', output_img)	# Show the stitched image
cv2.imwrite(out_filename, output_img) # Save the stitched image
cv2.waitKey(0) #Wait for key-press
