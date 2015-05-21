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
filename1='img1.jpg'
filename2='img2.jpg'
filename3='matched12.txt'
filename4="matched12.jpg"
#----------------------------------------------------------------------------------------------#
#Parameters used in the method

surf_th=1500  
score_th=0.25 
ratio_th=0.7 

#----------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#	
# This function takes in two sets of corner points and their feature descriptors and returns the matches found
def find_matches(cp1, cp2, fd1, fd2, matches, t, r):
	for loopVar1 in range (0, len(cp1)):			# For each corner point in image 1
		min_score=1e+10
		sec_min=1e+10
		point1=cp1[loopVar1][0]				# Get the point
		f1=fd1[loopVar1]				# Get the feature descriptor
		flag=0	
		for loopVar2 in range(0, len(cp2)):		# For each corner point in image 2
			point2=cp2[loopVar2][0]			# Get the point
			f2=fd2[loopVar2]			# Get the feature descriptor
			score=sqrt(sum(square(subtract(f1,f2))))	# Find the euclidean distance between two descriptors
			#print score		
			if score > t:				# Apply score threshold and ratio threshold
				continue
			else:
				if score < min_score:
					sec_min=min_score
					min_score=score
				elif (score > min_score)&(score < sec_min):
					sec_min=score
				else:
					continue
			ratio_of_score=min_score/sec_min
			if ratio_of_score > r:
				continue
			else:
				flag=1
				current_match=cp2[loopVar2][0]
		if flag==1:		
			matches.append([point1,current_match])		# Save the matched pair
			flag=0
			#print point1, current_match
		print loopVar1, len(cp1), len(cp2)
		#exit()
	else:
		pass		
		
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
# This function takes in two images and the matched set of points. Draws lines between matched points on a concatenated image
def draw_lines(img1, img2, matches):
	offset=img1.shape[1]
	#print offset
	
	final_image=array(concatenate((img1, img2), axis=1))		# Concatenate the two images
	
	for loopVar1 in range(0, len(matches)):		# For each pair of point in the matched set
		#print matches[loopVar1]
		cv2.line(final_image, (int(matches[loopVar1][0][0]), int(matches[loopVar1][0][1])), (int(matches[loopVar1][1][0])+offset, int(matches[loopVar1][1][1])), (0,0,255), 1)		# Draw a line
		cv2.circle(final_image, (int(matches[loopVar1][0][0]), int(matches[loopVar1][0][1])), 2, (255,0,0), -1) # Draw a point 
		cv2.circle(final_image, (int(matches[loopVar1][1][0])+offset, int(matches[loopVar1][1][1])), 2, (255,0,0), -1) # Draw a point
		
	return final_image
#---------------------------------------------------------------------------------------------#	
			
#---------------------------------------------------------------------------------------------#
# This function takes in two images and the matched set of points. Draws points at matched points 
def draw_points(img, points):
	offset=img.shape[1]
	for loopVar1 in range(0, len(points)):		# For each pair of point in the matched set
		cv2.circle(img, (int(points[loopVar1][0][0]), int(points[loopVar1][0][1])), 2, (255,0,0), -1) # Draw a point 
		
	return img
#--------------------------------------------------------------------------------------------#
#This function saves the correspondences to a text file to be read by RANSAC program 
def save_matches(filename, matches):
	fo = open(filename, 'w', 0)
	for loopVar1 in range(0, len(matches)):
		fo.write(str(matches[loopVar1][0][0]))
		fo.write('\t')
		fo.write(str(matches[loopVar1][0][1]))
		fo.write('\t')
		fo.write(str(matches[loopVar1][1][0]))
		fo.write('\t')
		fo.write(str(matches[loopVar1][1][1]))
		if loopVar1 !=len(matches)-1:
			fo.write('\n')
	fo.close()	
						
#---------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------#
#This function reads the matches saved in a text file. Not needed by this program. Used for debugging 
def readmatches(filename):    
    f = open(filename).read()
    rows = []
    for line in f.split('\n'):
        rows.append(line.split('\t'))
    
    for loopVar1 in range(0, len(rows)):
    	for loopVar2 in range(0, len(rows[loopVar1])):
    		rows[loopVar1][loopVar2]=float(rows[loopVar1][loopVar2])   
    return rows 
#--------------------------------------------------------------------------------------------# 

# Load image 1
orig_img_1 = cv2.imread(filename1,1)
input_img_1 = cv.LoadImage(filename1,0)

# Load image 2
orig_img_2 = cv2.imread(filename2,1)
input_img_2 = cv.LoadImage(filename2,0)

#---------------------------------------------------------------------------------------------#
# Use OpenCV's built in SURF function to get the corner points and their descriptors
(points1, desc1)=cv.ExtractSURF(input_img_1, None, cv.CreateMemStorage(), (0, surf_th, 3, 1))
(points2, desc2)=cv.ExtractSURF(input_img_2, None, cv.CreateMemStorage(), (0, surf_th, 3, 1))


matches=[]
find_matches(points1, points2, desc1, desc2, matches, score_th, ratio_th) # Use the descriptors to find the matches
print len(matches)
matched_image=draw_lines(orig_img_1, orig_img_2, matches) # Draw lines between matched points
cv2.imshow('final_matched', matched_image) # Show the matched image
cv2.imwrite(filename4, matched_image)  # Save the matched image
save_matches(filename3, matches) # Save the matches in a text file
#matches=readmatches(filename3)
#print matches
print len(matches) # print the number of matched points
cv2.waitKey(0) #Wait for key-press
