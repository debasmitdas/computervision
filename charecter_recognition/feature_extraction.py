# Importing libraries
import cv2
import cv
from math import *
from numpy import *
from sympy import Symbol,cos,sin
from operator import itemgetter
from operator import *
from numpy.linalg import *
import time
import ctypes

# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

#----------------------------------------------------------------------------------------------#
#Filenames
filename1='Image3_letter_'
filename2='Image3_letter_corner_'
filename3='Image3_Extracted_Features.txt'
#----------------------------------------------------------------------------------------------#
#Parameters used in the method
sigma=1.0
ws=3
wh=int(5*sigma)
wr=wh

k=0.04
N=13
#----------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#This function takes in dx and dy images to find the C matrix and hence corner strength 
def find_corner_strength(dx_img, dy_img, k, wh):
	img_height=dx_img.height
	img_width=dx_img.width
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
		#print "corner_strength", loopVar1
	return corner_strength_img
#---------------------------------------------------------------------------------------------#	
	
#---------------------------------------------------------------------------------------------#
# This function takes in the corner strength values for the whole image and applies threshold and non-maximum suppresion	
def non_max_supress(orig_img, corner_img, wr, th_r):
	img_height=corner_img.shape[0]
	img_width=corner_img.shape[1]	
	center_of_op=(wr/2)
	corner_counts=0
	corner_strengths=[]
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
			if max_corner_strength==corner_img.item(loopVar1,loopVar2):
				for loopVar3 in range(-center_of_op, center_of_op+1):	# Loop through the window
					for loopVar4 in range(-center_of_op, center_of_op+1):	
						if ((loopVar1+loopVar3)>=0 and (loopVar1+loopVar3)<img_height) and ((loopVar2+loopVar4)>=0 and (loopVar2+loopVar4)<img_width):
							if not(loopVar3==0 and loopVar4==0):
								corner_img.itemset(((loopVar1+loopVar3),(loopVar2+loopVar4)),0)
								
				if corner_img[loopVar1,loopVar2]<th_r:
					corner_img.itemset((loopVar1,loopVar2),0)
				else:
					corner_counts+=1
					#print corner_counts
					#print corner_img.item(loopVar1,loopVar2)
					#cv2.circle(orig_img, (loopVar2,loopVar1), 2, (255,0,0), -1)	
					corner_strengths.append([corner_img[loopVar1,loopVar2], [loopVar1,loopVar2]]) 			
		#print "non-maximum supression", loopVar1							
	return corner_strengths	
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
def find_feature_vector(best_corners, center_x, center_y):
	features=[]
	origin=[0, 0, 1]
	x_axis=[0, 1, 0]
	y_axis=[-1, 0, 0]
	angles_with_x_axis=[]
	angles_with_y_axis=[]
	for loopVar1 in range(len(best_corners)):
		temp=[best_corners[loopVar1][0]-center_x, best_corners[loopVar1][1]-center_y, 1]
		best_corners[loopVar1]=[temp[1], -temp[0], temp[2]]
		l=cross(origin, best_corners[loopVar1])
		angles_with_x_axis.append(arccos(dot(l,x_axis)/(norm(l)*norm(x_axis)))*360/(2*pi))
		angles_with_y_axis.append(arccos(dot(l,y_axis)/(norm(l)*norm(y_axis)))*360/(2*pi))
	for loopVar1 in range(len(best_corners)):
		if round(angles_with_y_axis[loopVar1]-angles_with_x_axis[loopVar1])==90:
			angles_with_x_axis[loopVar1]=[360-angles_with_x_axis[loopVar1], best_corners[loopVar1], loopVar1]
		elif round(angles_with_x_axis[loopVar1]+angles_with_y_axis[loopVar1])==270:
			angles_with_x_axis[loopVar1]=[360-angles_with_x_axis[loopVar1], best_corners[loopVar1], loopVar1]
		else:
			angles_with_x_axis[loopVar1]=[angles_with_x_axis[loopVar1], best_corners[loopVar1], loopVar1]
	#print angles_with_x_axis
	angles_with_x_axis=sorted(angles_with_x_axis, key=itemgetter(0))
	#print angles_with_x_axis			
	for loopVar1 in range(len(best_corners)):
		if (loopVar1+1) != len(best_corners):
			features.append(angles_with_x_axis[loopVar1+1][0]-angles_with_x_axis[loopVar1][0])
		else:
			features.append(360-(angles_with_x_axis[loopVar1][0]-angles_with_x_axis[0][0]))	
	return features
				
#---------------------------------------------------------------------------------------------#
#This function reads the matches saved in a text file.
def readfeatures(filename):    
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
def save_features(filename, features):
	fo = open(filename, 'w', 0)
	for loopVar1 in range(0, len(features)):
		for loopVar2 in range(0, len(features[loopVar1])):
			fo.write(str(features[loopVar1][loopVar2]))
			if loopVar2!=len(features[loopVar1])-1:
				fo.write('\t')
		#fo.write(str(loopVar1+1))
		if loopVar1!=len(features)-1:	
			fo.write('\n')
	fo.close()	
						
#---------------------------------------------------------------------------------------------#
# Load image 1 and get the corner points
feature_vectors=[]
for loopVar1 in range(0, 26):
	orig_img_1 = cv.LoadImage(filename1+str(loopVar1+1)+".jpg",1)
	input_img_1 = cv.LoadImage(filename1+str(loopVar1+1)+".jpg",0)
	dy_img_1=cv.CreateMat(orig_img_1.height, orig_img_1.width, cv.CV_32F)
	dx_img_1=cv.CreateMat(orig_img_1.height, orig_img_1.width, cv.CV_32F)
	cv.Sobel(input_img_1, dy_img_1, 0, 1, apertureSize=3)
	cv.Sobel(input_img_1, dx_img_1, 1, 0, apertureSize=3) 
	
	corner_img_1=find_corner_strength(dx_img_1, dy_img_1, k, wh)	# Find the corner strengths
	th_r=9e+8
	corner_strengths=non_max_supress(orig_img_1, corner_img_1, wr, th_r) # Apply threshold and non-maximum supression
	#print len(corner_strengths), len(corner_strengths)-N
	#print corner_strengths
	corner_strengths=sorted(corner_strengths, key=itemgetter(0))
	best_corners=[]
	for loopVar2 in range(len(corner_strengths)-1, len(corner_strengths)-1-N, -1):
		try:
			cv.Circle(orig_img_1, (corner_strengths[loopVar2][1][1],corner_strengths[loopVar2][1][0]), 2, (255,0,0), -1, lineType=8, shift=0) 
			best_corners.append([corner_strengths[loopVar2][1][0], corner_strengths[loopVar2][1][1]])	
			#print corner_strengths[loopVar2]
		except IndexError:
			pass
	#print best_corners, (orig_img_1.height)/2, (orig_img_1.width)/2		
	feature_vectors.append(find_feature_vector(best_corners, (orig_img_1.height)/2, (orig_img_1.width)/2))
	print feature_vectors[loopVar1], sum(feature_vectors[loopVar1])							
	cv.SaveImage(filename2+str(loopVar1+1)+".jpg", orig_img_1) #Save the result
save_features(filename3, feature_vectors)
read=readfeatures(filename3)
print read, len(read)	
cv2.waitKey(0) #Wait for key-press'''
