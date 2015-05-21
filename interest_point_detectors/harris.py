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
filename1='pic1.jpg'
filename2='pic2.jpg'
#----------------------------------------------------------------------------------------------#
#Parameters used in the method
sigma=2.6
ws=3
wh=int(5*sigma)
wr=wh

wncc=39  #47
tncc=0.95 #0.75
rncc=0.95 #0.8

wssd=53   # 45
tssd=100  # 1200
rssd=0.6  #0.9

k=0.04

#----------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#This function takes in dx and dy images to find the C matrix and hence corner strength 
def find_corner_strength(dx_img, dy_img, k, wh):
	img_height=dx_img.shape[0]
	img_width=dx_img.shape[1]
	center_of_op=(wh/2)
	corner_strength_img = zeros((img_height,img_width), float)
	dx_img_sqr=square(dx_img)		# Get square of dx 
	dy_img_sqr=square(dy_img)		# Get square of dy
	dxdy_img=multiply(dx_img,dy_img)	# Get product of dx and dy
	
	for loopVar1 in range(0, img_height):		# Loop for all pixels
		for loopVar2 in range(0, img_width):
			sum_dx_sqr=0.0
			sum_dy_sqr=0.0
			sum_dxdy=0.0
			for loopVar3 in range(-center_of_op, center_of_op+1):	# Loop through the window
				for loopVar4 in range(-center_of_op, center_of_op+1):
					if ((loopVar1+loopVar3)>=0 and (loopVar1+loopVar3)<img_height) and ((loopVar2+loopVar4)>=0 and (loopVar2+loopVar4)<img_width):
						# Find required summations
						sum_dx_sqr+=dx_img_sqr.item((loopVar1+loopVar3),(loopVar2+loopVar4))
						sum_dy_sqr+=dy_img_sqr.item((loopVar1+loopVar3),(loopVar2+loopVar4))
						sum_dxdy+=dxdy_img.item((loopVar1+loopVar3),(loopVar2+loopVar4))

			# Calculate corner strength and save it
			corner_strength=((sum_dx_sqr*sum_dy_sqr)-square(sum_dxdy)-k*square(sum_dx_sqr+sum_dy_sqr))/(wh*wh)
			corner_strength_img.itemset((loopVar1,loopVar2),corner_strength)	
		print "corner_strength", loopVar1
	return corner_strength_img
#---------------------------------------------------------------------------------------------#	
	
#---------------------------------------------------------------------------------------------#
# This function takes in the corner strength values for the whole image and applies threshold and non-maximum suppresion	
def non_max_supress(orig_img, corner_img, wr, th_r, corner_points):
	img_height=corner_img.shape[0]
	img_width=corner_img.shape[1]	
	center_of_op=(wr/2)
	corner_counts=0
	for loopVar1 in range(0, img_height):	# Loop for all pixels
		for loopVar2 in range(0, img_width):
			max_corner_strength=-1e+9
			for loopVar3 in range(-center_of_op, center_of_op+1):	# Loop through the window
				for loopVar4 in range(-center_of_op, center_of_op+1):	
					if ((loopVar1+loopVar3)>=0 and (loopVar1+loopVar3)<img_height) and ((loopVar2+loopVar4)>=0 and (loopVar2+loopVar4)<img_width):
						# Find the maximum corner strength within the window
						if corner_img.item((loopVar1+loopVar3),(loopVar2+loopVar4))>max_corner_strength:
							max_corner_strength=corner_img.item((loopVar1+loopVar3),(loopVar2+loopVar4))
			
			# Apply threshold and suppress non-maximum strength values 
			if max_corner_strength!=corner_img.item(loopVar1,loopVar2):
				corner_img.itemset((loopVar1,loopVar2),0)
			elif corner_img[loopVar1,loopVar2]<th_r:
				corner_img.itemset((loopVar1,loopVar2),0)
			else:
				corner_counts+=1
				print corner_counts
				#print corner_img.item(loopVar1,loopVar2)
				#cv2.circle(orig_img, (loopVar2,loopVar1), 2, (255,0,0), -1)
				corner_points.append([loopVar1,loopVar2])							
		print "non-maximum supression", loopVar1
									
	return orig_img	
#---------------------------------------------------------------------------------------------#
	
#---------------------------------------------------------------------------------------------#	
# This function takes in two sets of corner points and a type of metric (ncc or ssd) and return the matches found
def find_matches(cp1, cp2, metric, matches, w, t, r):
	img_1_height=input_img_1.shape[0]
	img_1_width=input_img_1.shape[1]
	img_2_height=input_img_2.shape[0]
	img_2_width=input_img_2.shape[1]
	center_of_op=(w/2)
				
	if metric=="ncc":					# If metric to be used is "NCC"
		for loopVar1 in range (0, len(cp1)):		# For each corner point in image 1
			max_ncc=0
			sec_max=0
			f1=[]
			m1 = zeros((w,w),float)
			point1=cp1[loopVar1]
			for loopVar3 in range(-center_of_op, center_of_op+1): # Find f1 and m1
				temp1=[]
				for loopVar4 in range(-center_of_op, center_of_op+1):
					if ((point1[0]+loopVar3)>=0 and (point1[0]+loopVar3)<img_1_height) and ((point1[1]+loopVar4)>=0 and (point1[1]+loopVar4)<img_1_width):
						temp1.append(input_img_1[(point1[0]+loopVar3),(point1[1]+loopVar4)])
				if len(temp1)==w:
					f1.append(temp1)
			if (len(f1)!=w):
				continue
				
			m1.fill(mean(f1))
			f1_m1=subtract(f1,m1)
			sum_of_squares_1=sum(square(f1_m1))
			flag=0				
			for loopVar2 in range(0, len(cp2)):	# For each corner point in image 2
				f2=[]
				m2 = zeros((w,w),float)
				point2=cp2[loopVar2]
				for loopVar3 in range(-center_of_op, center_of_op+1): # Find f2 and m2
					temp2=[]
					for loopVar4 in range(-center_of_op, center_of_op+1):
						if ((point2[0]+loopVar3)>=0 and (point2[0]+loopVar3)<img_2_height) and ((point2[1]+loopVar4)>=0 and (point2[1]+loopVar4)<img_2_width):
							temp2.append(input_img_2[(point2[0]+loopVar3),(point2[1]+loopVar4)])
					if len(temp2)==w:
						f2.append(temp2)
				if len(f2)!=w:		
					continue
				
				m2.fill(mean(f2))				# Calculate required products and sums
				f2_m2=subtract(f2,m2)	
				prod=multiply(f1_m1,f2_m2)
				sum_of_prod=sum(prod)
				sum_of_squares_2=sum(square(f2_m2))
				ncc=sum_of_prod/sqrt(sum_of_squares_1*sum_of_squares_2) # Calculate NCC
								
				if ncc < t:					# Apply score threshold and ratio threshold
					continue
				else:
					if ncc > max_ncc:
						sec_max=max_ncc
						max_ncc=ncc
					elif (ncc < max_ncc)&(ncc > sec_max):
						sec_max=ncc
					else:
						continue
				ratio_of_ncc=sec_max/max_ncc
				if ratio_of_ncc > r:
					continue
				else:
					flag=1
					current_match=cp2[loopVar2]
			if flag==1:		
				matches.append([cp1[loopVar1],current_match])	# Save the matched pair
			print loopVar1
							
	elif metric=="ssd":					# If metric to be used is "SSD"
		for loopVar1 in range (0, len(cp1)):		# For each corner point in image 1
			min_ssd=1e+10
			sec_min=1e+10
			f1=[]
			point1=cp1[loopVar1]
			for loopVar3 in range(-center_of_op, center_of_op+1):	# Find f1
				temp1=[]
				for loopVar4 in range(-center_of_op, center_of_op+1):
					if ((point1[0]+loopVar3)>=0 and (point1[0]+loopVar3)<img_1_height) and ((point1[1]+loopVar4)>=0 and (point1[1]+loopVar4)<img_1_width):
						temp1.append(input_img_1[(point1[0]+loopVar3),(point1[1]+loopVar4)])
				if len(temp1)==w:
					f1.append(temp1)
			if (len(f1)!=w):
				continue
			flag=0	
			for loopVar2 in range(0, len(cp2)):			# For each corner point in image 2
				f2=[]
				point2=cp2[loopVar2]
				for loopVar3 in range(-center_of_op, center_of_op+1):	# Find f2
					temp2=[]
					for loopVar4 in range(-center_of_op, center_of_op+1):
						if ((point2[0]+loopVar3)>=0 and (point2[0]+loopVar3)<img_2_height) and ((point2[1]+loopVar4)>=0 and (point2[1]+loopVar4)<img_2_width):
							temp2.append(input_img_2[(point2[0]+loopVar3),(point2[1]+loopVar4)])
					if len(temp2)==w:
						f2.append(temp2)
				if len(f2)!=w:		
					continue
					
				ssd=sum(square(subtract(f1,f2)))/(w*w)		# Calculate SSD
					
				if ssd > t:				# Apply score threshold and ratio threshold
					continue
				else:
					if ssd < min_ssd:
						sec_min=min_ssd
						min_ssd=ssd
					elif (ssd > min_ssd)&(ssd < sec_min):
						sec_min=ssd
					else:
						continue
				ratio_of_ssd=min_ssd/sec_min
				if ratio_of_ssd > r:
					continue
				else:
					flag=1
					current_match=cp2[loopVar2]
			if flag==1:		
				matches.append([cp1[loopVar1],current_match])	 # Save the matched pair
			
			print loopVar1
	else:
		pass		
		
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
# This function takes in two images and the matched set of points. Draws lines between matched points on a concatenated image
def draw_lines(img1, img2, matches):
	offset=img1.shape[1]
	final_image=array(concatenate((img1, img2), axis=1))	# Concatenate the two images
	
	for loopVar1 in range(0, len(matches)):		# For each pair of point in the matched set
		cv2.line(final_image, (matches[loopVar1][0][1], matches[loopVar1][0][0]), (matches[loopVar1][1][1]+offset, matches[loopVar1][1][0]), (255*(loopVar1%4),255*((loopVar1+1)%4),255*((loopVar1+2)%4)), 1)	# Draw a line
		cv2.circle(final_image, (matches[loopVar1][0][1], matches[loopVar1][0][0]), 2, (255,0,0), -1) # Draw a point 
		cv2.circle(final_image, (matches[loopVar1][1][1]+offset, matches[loopVar1][1][0]), 2, (255,0,0), -1) # Draw a point
		
	cv2.imshow('final_matched', final_image)
	cv2.imwrite("ncc_39_0.95_scale_2_6.jpg", final_image)	# Save the resultant image
	cv2.waitKey(0) #Wait for key-press
#---------------------------------------------------------------------------------------------#	
									
#---------------------------------------------------------------------------------------------#
# Load image 1 and get the corner points
orig_img_1 = cv2.imread(filename1,1)
input_img_1 = cv2.imread(filename1,0)
dy_img_1=cv2.Sobel(input_img_1, -1, 0, 1)	# Apply sobel to get dx and dy
dx_img_1=cv2.Sobel(input_img_1, -1, 1, 0)
dx_img_1=dx_img_1.astype(float32)
dy_img_1=dy_img_1.astype(float32)
corner_img_1=find_corner_strength(dx_img_1, dy_img_1, k, wh)	# Find the corner strengths
th_r=2e+9
corner_points_1=[]
orig_img_1=non_max_supress(orig_img_1, corner_img_1, wr, th_r, corner_points_1) # Apply threshold and non-maximum supression


# Load image 2 and get the corner points
orig_img_2 = cv2.imread(filename2,1)
input_img_2 = cv2.imread(filename2,0)
dy_img_2=cv2.Sobel(input_img_2, -1, 0, 1)	# Apply sobel to get dx and dy
dx_img_2=cv2.Sobel(input_img_2, -1, 1, 0)
dx_img_2=dx_img_2.astype(float32)
dy_img_2=dy_img_2.astype(float32)
corner_img_2=find_corner_strength(dx_img_2, dy_img_2, k, wh)	# Find the corner strengths
th_r=2e+9
corner_points_2=[]
orig_img_2=non_max_supress(orig_img_2, corner_img_2, wr, th_r, corner_points_2) # Apply threshold and non-maximum supression
#---------------------------------------------------------------------------------------------#

matches=[]
find_matches(corner_points_1, corner_points_2, "ncc", matches, wncc, tncc, rncc) # Find the matches usng ncc 
draw_lines(orig_img_1, orig_img_2, matches)	# Draw lines between matched points
print len(matches)				# print the number of matched points
print wncc, tncc, rncc				# print the parameters
find_matches(corner_points_1, corner_points_2, "ssd", matches, wssd, tssd, rssd) # Find the matches usng ssd
draw_lines(orig_img_1, orig_img_2, matches)	# Draw lines between matched points
print len(matches)				# print the number of matched points
print wssd, tssd, rssd				# print the parameters
