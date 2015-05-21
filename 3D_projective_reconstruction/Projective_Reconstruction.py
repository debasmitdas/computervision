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
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

filename='Dataset 5/'  # Dataset Used
#----------------------------------------------------------------------------------------------#
#Parameters used in the method
SEARCH_WINDOW=3		# Searching window of rows +/- 3 rows

# Canny parameters
canny_lowThreshold=70
canny_threshold_ratio=2
canny_kernel_size=3

# NCC parameters
w=5			# NCC neighbor window size
center_of_op=(w/2)
ncc_th=0.9
#ncc_r_th=0.8

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
	cv2.imwrite(filename+"matched_points.jpg", final_image)		# Save the resultant image	
	return final_image

#---------------------------------------------------------------------------------------------#	
#This function saves the correspondences to a text file to be read by 3D plotting routine 
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
						
#--------------------------------------------------------------------------------------------#
#This function reads the matrices or points saved in a text file
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
# This function takes in a binary edge image and extracts the interest points just by checking if the pixel value is 1 or not
def get_points(image, x_range, y_range):
	points=[]
	for loopVar1 in range(y_range[0], y_range[1]+1):	# For all pixels within specified window
		for loopVar2 in range(x_range[0], x_range[1]+1):
			if image[loopVar1, loopVar2]==1:		# Check if the pixel value is 1
				points.append([loopVar2, loopVar1])	# If yes, save the point
	return points	
			
#---------------------------------------------------------------------------------------------#
# This funtion takes in an image, a point and window size to find the f and m values in the neighbor -> used by NCC
def find_fm_neighbor(point, image, w):
	f=[]					# Initialize f and m matrices
	m = zeros((w,w),float)		
	center_of_op=(w/2)	
	image_height=image.height		# Get size of image
	image_width=image.width
	for loopVar3 in range(-center_of_op, center_of_op+1): # Find f and m by looping through the window
		temp1=[]
		for loopVar4 in range(-center_of_op, center_of_op+1):
			if ((point[0]+loopVar3)>=0 and (point[0]+loopVar3)<image_width) and ((point[1]+loopVar4)>=0 and (point[1]+loopVar4)<image_height):
				temp1.append(image[(point[1]+loopVar4), (point[0]+loopVar3)])
		if len(temp1)==w:
			f.append(temp1)
	m.fill(mean(f))				# Fill in mean matrix m
	fm=subtract(f,m)			# Subtract mean matrix from f matric
	return fm				# Return the (f-m) window

#---------------------------------------------------------------------------------------------#
# This function takes in two points and finds the ncc between those two
def find_ncc(point1, point2): 
	fm1=find_fm_neighbor(point1, input_img_1, w)	# Finds (f-m) window for point 1
	fm2=find_fm_neighbor(point2, input_img_2, w)	# Finds (f-m) window for point 2
	sum_of_squares_1=sum(square(fm1))		# Find sum of squares
	sum_of_squares_2=sum(square(fm2))
	prod=multiply(fm1,fm2)				# Find product of (f-m) windows
	sum_of_prod=sum(prod)				# Find sum of product
	ncc=sum_of_prod/sqrt(sum_of_squares_1*sum_of_squares_2) # Calculate NCC
	return ncc					# Return the NCC score

#---------------------------------------------------------------------------------------------#
# This function takes in two sets of points of two images and ncc parameters. It finds the matched points using NCC metric.
def find_correspondences(points1, points2, ncc_th, r):
	matches=[]
	for loopVar1 in range(len(points1)):		# For each point in image 1
		max_ncc=0		
		flag=0
		for loopVar2 in range(len(points2)):	# For each point in image 2
			if abs(points2[loopVar2][1]-points1[loopVar1][1])<SEARCH_WINDOW: # If the point 2 is within search window
				ncc=find_ncc(points1[loopVar1], points2[loopVar2])	# Find NCC
				if ncc > ncc_th:				# If NCC exceeds threshold
					if ncc > max_ncc:			# Check if its greater than current maximum score
						max_ncc=ncc			# If yes, update the maximum and update the best point
						flag=1
						current_match=points2[loopVar2]
						
		if flag==1:							# If there is a match found
			matches.append([points1[loopVar1],current_match])	# Save the matched pair
		print loopVar1
	return matches						# Return the set of matched points

#---------------------------------------------------------------------------------------------#	
# This function implements triangulation. It takes in set of image points and project matrices. Finds the correspondings world points
def ProjectToWorld(op, P1, P2):
	world_points=[]
	for loopVar1 in range(len(op)):			# For all image correspondences
		A=matrix(zeros((4,4)))				# Find A matrix
		A[0,:]=(op[loopVar1, 0]*P1[2,:])-(P1[0,:])
		A[1,:]=(op[loopVar1, 1]*P1[2,:])-(P1[1,:])
		A[2,:]=(op[loopVar1, 2]*P2[2,:])-(P2[0,:])
		A[3,:]=(op[loopVar1, 3]*P2[2,:])-(P2[1,:])
	
		world_X=transpose(matrix((linalg.svd(transpose(matrix(A))*matrix(A))[2][3]).tolist()[0])) # Find world point
		world_X=world_X/world_X[3,0]		
		world_points.append([world_X[0, 0], world_X[1, 0], world_X[2, 0]]) 		# Save the world points
	return world_points									# Return them

#---------------------------------------------------------------------------------------------#
# This function takes in world points and draws 3D plot using plotting tools
def draw3D(world_points):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')			# Create a figure and add 3D sub plot
	x = []
	y = []
	z = []
	count = 0
	for loopVar1 in range(len(world_points)):			# For all world points
		if abs(world_points[loopVar1][2]) < 20:			# get rid of outliers!
			x.append(world_points[loopVar1][0])		# Extract x, y and z coordinates
			y.append(world_points[loopVar1][1])
			z.append(world_points[loopVar1][2])
			count+=1
	ax.scatter(x, y, z, zdir='z', s=1)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()

#---------------------------------------------------------------------------------------------#
# Main Code starts
# Load both the images in color as well as gray scale
										# Load image 1
orig_img_1 = cv2.imread(filename+'Pic_1_corrected.jpg',1)
input_img_1 = cv.LoadImage(filename+'Pic_1_corrected.jpg',0)

										# Load image 2
orig_img_2 = cv2.imread(filename+'Pic_2_corrected.jpg',1)
input_img_2 = cv.LoadImage(filename+'Pic_2_corrected.jpg',0)

										# Apply Canny Edge detector for Image 1
detected_edges=cv.CreateImage((orig_img_1.shape[1], orig_img_1.shape[0]), cv.IPL_DEPTH_8U, 1)
cv.Canny(input_img_1, detected_edges, canny_lowThreshold, canny_lowThreshold*canny_threshold_ratio, canny_kernel_size ) # Apply Canny detector	
cv.SaveImage(filename+'Pic_1_edge.jpg', detected_edges)
edge_image_1 = cv2.imread(filename+'Pic_1_edge.jpg',0)
										# Apply Canny Edge detector for Image 2
detected_edges=cv.CreateImage((orig_img_2.shape[1], orig_img_2.shape[0]), cv.IPL_DEPTH_8U, 1)
cv.Canny(input_img_2, detected_edges, canny_lowThreshold, canny_lowThreshold*canny_threshold_ratio, canny_kernel_size ) # Apply Canny detector	
cv.SaveImage(filename+'Pic_2_edge.jpg', detected_edges)
edge_image_2 = cv2.imread(filename+'Pic_2_edge.jpg',0)

points1=get_points(edge_image_1, (170, 600), (5, 430))				# Extract interest points in Image 1
points2=get_points(edge_image_2, (115, 550), (5, 430))				# Extract interest points in Image 2

print len(points1), len(points2), points1[len(points1)-1]			
matches=find_correspondences(points1, points2, ncc_th, ncc_r_th)		# Find matches using NCC method
print len(matches)	
draw_lines(orig_img_1, orig_img_2, matches)					# Draw lines between matched points in 2D images
save_matches(filename+'matches.txt', matches)					# Save the matches

correspondences=matrix(readmatches(filename+'matches.txt'))			# Read the matches
P1=matrix(readmatches(filename+'P1.txt'))					# Read P1 and P2 matrices
P2=matrix(readmatches(filename+'P2.txt'))	
print len(correspondences), P1, P2
world_points=ProjectToWorld(correspondences, P1, P2)				# Project points to world coordinates
print len(world_points)
draw3D(world_points)								# Draw 3D plots of the world coordinates

'''										# SURF Code (TRIED - NOT USED FINALLY)
#---------------------------------------------------------------------------------------------#
# Use OpenCV's built in SURF function to get the corner points and their descriptors
(points1, desc1)=cv.ExtractSURF(input_img_1, None, cv.CreateMemStorage(), (0, surf_th, 3, 1))
(points2, desc2)=cv.ExtractSURF(input_img_2, None, cv.CreateMemStorage(), (0, surf_th, 3, 1))


for loopVar1 in range(len(points1)):
	points1[loopVar1]=points1[loopVar1][0]
for loopVar1 in range(len(points2)):
	points2[loopVar1]=points2[loopVar1][0]

points1=sorted(points1, key=itemgetter(1))
points2=sorted(points2, key=itemgetter(1))
print points2
exit()
matches=[]
find_matches(points1, points2, desc1, desc2, matches, score_th, ratio_th) # Use the descriptors to find the matches
draw_lines(orig_img_1, orig_img_2, matches)	# Draw the lines between matches points
print len(matches)				# print the number of matched points
print len(points1), len(points2)'''
