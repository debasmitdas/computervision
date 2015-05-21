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
from matplotlib import pyplot as plt

# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

#----------------------------------------------------------------------------------------------#
#Filenames
filename='Dataset1/Pic_'
	
#----------------------------------------------------------------------------------------------#
# Parameters used
canny_lowThreshold=210
canny_threshold_ratio=2
canny_kernel_size=3

HT_VOTES_THRESHOLD=30
HT_MIN_LINE_LENGTH=30
HT_MAX_DIST_BW_LINES=600

CALIB_SQUARE_SIZE=2.44  # cm

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
def save_corners(filename, features):
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

#---------------------------------------------------------------------------------------------#	
def find_corner(l1,l2):
	X=[l1[0][0], l1[0][1], 1]
	Y=[l1[1][0], l1[1][1], 1]
	line1=cross(X,Y)
	X=[l2[0][0], l2[0][1], 1]
	Y=[l2[1][0], l2[1][1], 1]
	line2=cross(X,Y)
	temp_l1=[line1[0], line1[1]]
	temp_l2=[line2[0], line2[1]]
	angle=arccos(dot(temp_l1,temp_l2)/(norm(temp_l1)*norm(temp_l2)))*360/(2*pi)
	if (angle>30)and(angle<150):
		corner=cross(line1,line2)
	else:
		corner=[0, 0, 0]
	return corner
#---------------------------------------------------------------------------------------------#	

def find_line_type(line, return_type):
	p1=[line[0][0], line[0][1], 1]
	p2=[line[1][0], line[1][1], 1]
	line=cross(p1,p2)
	temp_line=[line[0], line[1]]
	x_axis=[0, 1]
	y_axis=[-1, 0]
	angle_x=arccos(dot(temp_line,x_axis)/(norm(temp_line)*norm(x_axis)))*360/(2*pi)
	#angle_y=arccos(dot(temp_line,y_axis)/(norm(temp_line)*norm(y_axis)))*360/(2*pi)
	#print angle_x, angle_y
	x_axis=[0, 1, 0]
	y_axis=[-1, 0, 0]
	y_intercept=cross(line,y_axis)
	if angle_x>45:
		if return_type=='type':
			return 0  # vertical line
		if return_type=='intercept':
			x_intercept=cross(line,x_axis)
			x_intercept=x_intercept/float(x_intercept[2])
			return x_intercept[0]	
	else:	
		if return_type=='type':
			return 1  # horizontal line
		if return_type=='intercept':
			y_intercept=cross(line,y_axis)
			y_intercept=y_intercept/float(y_intercept[2])
			return y_intercept[1]		
#---------------------------------------------------------------------------------------------#			
# Main code
# Load input image

for loopVar1 in range(1, 41):
	orig_img = cv.LoadImage(filename+str(loopVar1)+'.jpg', 1)
	detected_edges=cv.CreateImage((orig_img.width, orig_img.height), orig_img.depth, 1)
	cv.Canny(orig_img, detected_edges, canny_lowThreshold, canny_lowThreshold*canny_threshold_ratio, canny_kernel_size ) # Apply Canny detector	
	cv.SaveImage(filename+str(loopVar1)+'_edge.jpg', detected_edges)
	lines=cv.HoughLines2(detected_edges, cv.CreateMemStorage(), cv.CV_HOUGH_PROBABILISTIC, 1, cv.CV_PI/180, HT_VOTES_THRESHOLD, HT_MIN_LINE_LENGTH, HT_MAX_DIST_BW_LINES)
	corners=[]
	filtered_lines=[]
	for loopVar2 in range(len(lines)):
		for loopVar3 in range(len(lines)):
			corner=find_corner(lines[loopVar2], lines[loopVar3])
			if corner[2]!=0:
				corner=corner/corner[2]
			else:
				corner=[0,0,0]	
			if (corner[0]>0 and corner[0]<orig_img.width) and (corner[1]>0 and corner[1]<orig_img.height):
				flag=0
				for item in corners:
					if sqrt(sum(square(subtract([corner[0], corner[1]], item))))<20:
						flag=1
						break
				if flag==0:		
					corners.append([corner[0], corner[1]])
					if not(((lines[loopVar2],0) in filtered_lines)or((lines[loopVar2],1) in filtered_lines)):
						filtered_lines.append((lines[loopVar2],find_line_type(lines[loopVar2], 'type')))
					if not(((lines[loopVar3],0) in filtered_lines)or((lines[loopVar3],1) in filtered_lines)):	
						filtered_lines.append((lines[loopVar3],find_line_type(lines[loopVar3], 'type')))

	vertical_lines=[]
	horizontal_lines=[]							
	for loopVar2 in range(len(filtered_lines)):
		filtered_lines[loopVar2]=(filtered_lines[loopVar2], find_line_type(filtered_lines[loopVar2][0], 'intercept'))
		if filtered_lines[loopVar2][0][1]==1:
			horizontal_lines.append(filtered_lines[loopVar2])
		else:
			vertical_lines.append(filtered_lines[loopVar2])
	horizontal_lines=sorted(horizontal_lines, key=itemgetter(1))
	vertical_lines=sorted(vertical_lines, key=itemgetter(1))
	
	
	sorted_corners=[]
	for h_line in horizontal_lines:
		for v_line in vertical_lines:
			 corner=find_corner(h_line[0][0],v_line[0][0])
			 if corner[2]!=0:
				corner=corner/float(corner[2])
			 else:
				corner=[0,0,0]
			 if (corner[0]>0 and corner[0]<orig_img.width) and (corner[1]>0 and corner[1]<orig_img.height):
				flag=0
				for item in sorted_corners:
					if sqrt(sum(square(subtract([corner[1], corner[0]], item))))<20:
						flag=1
						break
				if flag==0:		
					sorted_corners.append([corner[1], corner[0]])
						
	
	print len(sorted_corners), len(filtered_lines), len(horizontal_lines), len(vertical_lines), loopVar1
	for (((x,y),t),c) in filtered_lines:
		if t==1:
			#cv.Line(orig_img, x, y, cv.Scalar(0,0,255), 1, cv.CV_AA) # Horzontal lines in Red
			#cv.Line(orig_img, x, (0, int(c)), cv.Scalar(0,0,255), 1, cv.CV_AA) # Horzontal lines in Red
			cv.Circle(orig_img, (0, int(c)), 2, (0,0,0), -1) # Draw intercept point	
		else:
			#cv.Line(orig_img, x, y, cv.Scalar(0,255,0), 1, cv.CV_AA) # Verical lines in green
			#cv.Line(orig_img, x, (int(c), 0), cv.Scalar(0,255,0), 1, cv.CV_AA) # Verical lines in green
			cv.Circle(orig_img, (int(c), 0), 2, (0,0,0), -1) # Draw intercept point
	font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, thickness=1)		
	for index, corner in enumerate(sorted_corners):
		cv.Circle(orig_img, (int(corner[1]), int(corner[0])), 2, (255,0,0), -1) # Draw a point
		#cv.PutText(orig_img, str(index+1), (int(corner[1]), int(corner[0])), font, (255,255,0))		
	#cv.ShowImage('Edges', detected_edges) # Show the matched image
	cv.SaveImage(filename+str(loopVar1)+'_just_corners.jpg', orig_img) #Save the result
	save_corners(filename+str(loopVar1)+'_corners.txt', sorted_corners)
	sorted_corners=readcorners(filename+str(loopVar1)+'_corners.txt')
	#print sorted_corners
	print len(sorted_corners), 'verification'
	#cv2.waitKey(0) #Wait for key-press'''
	
world_coordinates=[]
for loopVar1 in range(len(horizontal_lines)):
	for loopVar2 in range(len(vertical_lines)):
		world_coordinates.append([(loopVar1*CALIB_SQUARE_SIZE),(loopVar2*CALIB_SQUARE_SIZE)])
save_corners('world_coordinates.txt', world_coordinates)
world_coordinates=readcorners('world_coordinates.txt')		
print len(world_coordinates) 		
