# Importing libraries
import cv2
import cv
from math import *
from numpy import *
from sympy import Symbol,cos,sin
import random as rand

from operator import *
from numpy.linalg import *
import time
import ctypes

# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

#----------------------------------------------------------------------------------------------#
#Filenames
filename1='matched67.txt'
filename2='homography67.txt'
filename3='matched67.jpg'
filename4='ransac67.jpg'
#----------------------------------------------------------------------------------------------#
#Parameters used in the method
sigma=3 # Noise
delta=3*sigma # 
p=0.99
e=0.20
n=8
N=log(1-p)/log(1-pow((1-e),n))

#----------------------------------------------------------------------------------------------#
# This function takes in correspondences between two images, minimum number of inliers and number of iterations. It applies RANSAC algorithm and returns the best homography between the two images
def ransac(matched_points, delta, M, N):
	min_error=1e+10
	best_inlier_set=[]
	best_H=[]
	
	for loopVar1 in range(0, N):	# for N iterations
		sampled_matches = rand.sample(matched_points, n) # Pick n random points
		H=compute_homography(sampled_matches) # compute_homography
		#print H
		inlier_set, inlier_count, error = evaluate_homography(H, matched_points, delta) # Evaluate and get the inliers
		
		if inlier_count >= M:	# If number of inliers is more than M
			if error<min_error: # If the current error is less than previous minimum error
				min_error=error
				#print H
				#print inlier_count
				#print min_error
				best_inlier_set=inlier_set  # Remember the homography and its inlier set
				best_H=H
				
	# Get the best homography using the best inlier set			
	if len(best_inlier_set)!=0:			 
		final_H=compute_homography(best_inlier_set) # Compute best homography
		print final_H
		return final_H, best_inlier_set # Return the best homography
	else:
		return 0, best_inlier_set # Return 0 if there is no best homography		
#--------------------------------------------------------------------------------------------#
# This function takes in a homography, all correspondences and the threshold for a correspondence to be considered inlier. It returns the inlier set and the number of inliers
def evaluate_homography(H, matched_points, delta):
	inlier_count=0
	inlier_set=[]
	sum_distances=0 
	for loopVar1 in range(0, len(matched_points)):	# For all correspondences
		X=[[matched_points[loopVar1][0]],[matched_points[loopVar1][1]],[1]] # Get the point in image 1
		X_bar=H*X								# Apply homography
		X_bar=[float(X_bar[0][0]/X_bar[2][0]), float(X_bar[1][0]/X_bar[2][0])]	# Get the physical coordinates
		X_bar_orig=[matched_points[loopVar1][2], matched_points[loopVar1][3]]	# Get the observed physical coordinates

		distance=sqrt(sum(square(subtract(X_bar,X_bar_orig))))	# Calculate the error (distance)
		#print distance
		
		if distance <= delta:		# If error is less than the threshold
			inlier_set.append(matched_points[loopVar1])	# Save the correspondence as an inlier
			inlier_count+=1					# Increment the inlier count
			sum_distances+=distance				# Accumulate the error
	#print inlier_count
	#exit()
		
	return inlier_set, inlier_count, sum_distances	# Return the inlier set and corresponding error
#--------------------------------------------------------------------------------------------#
	
#--------------------------------------------------------------------------------------------#
# This function takes in a set of correspondences and finds out the homography using linear least squares method
def compute_homography(sampled_matches):
	w=1
	w_bar=1
	#Calculate the H Matrix from AH=b
	A=[] # A Matrix
	b=[] # b Matrix

	#This loop fills in Matrix A and y
	for loopVar1 in range(0,len(sampled_matches)):	
		x=sampled_matches[loopVar1][0]
		y=sampled_matches[loopVar1][1]
		x_bar=sampled_matches[loopVar1][2]
		y_bar=sampled_matches[loopVar1][3]
		A.append([0, 0, 0, (-w_bar*x), (-w_bar*y), (-w_bar*w), (y_bar*x), (y_bar*y)])
		A.append([(w_bar*x), (w_bar*y), (w_bar*w), 0, 0, 0, (-x_bar*x), (-x_bar*y)])
		b.append([(-y_bar*w)])
		b.append([(x_bar*w)])

	#Find out least squares solution and fill in H matrix
	x=linalg.lstsq(A, b)[0]
	H=matrix([[x[0][0],x[1][0],x[2][0]],[x[3][0],x[4][0],x[5][0]],[x[6][0],x[7][0],1]])
	return H	
#--------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------#
# This function implements the dog-leg non-linear optimization method. It takes in the initial H, best inlier set and returns the optimal H
def dogleg_refinement(H, best_inlier_set):
	r=2.0
	X=[]
	for loopVar1 in range(0, len(best_inlier_set)):	# Get the X vector (Physical coordinates of image 1 matches)
		X.append(best_inlier_set[loopVar1][2])
		X.append(best_inlier_set[loopVar1][3])
	stop=0
	while stop==0:				# Repeat until stop condition is met
		F=findF(H, best_inlier_set)		# Find F vector (Estimated physical coordinates of image 2 matches
		C=sum(square(subtract(X,F)))		# Find the geometric error 
		E=transpose(matrix(subtract(X,F)))	# Fidn the error vector
		J=findJ(H, best_inlier_set)		# Find the jacobian for given homography and inlier set

		# Calculate the steps for Gradient Descent and Gauss Newton 
		delta_GD=(sqrt(sum(square(transpose(J)*E)))/sqrt(sum(square(J*transpose(J)*E))))*transpose(J)*E
		delta_GN=inv(transpose(J)*J)*transpose(J)*E
		#print delta_GD, type(delta_GD), len(delta_GD)
		#print delta_GN, type(delta_GN), len(delta_GN)
		
		# Checks for three different cases of dog-leg
		if sqrt(sum(square(delta_GN)))<r:
			delta_final=delta_GN
			print 'GN'
		elif (sqrt(sum(square(delta_GD)))<=r) and (sqrt(sum(square(delta_GN)))>r):
			beta=find_beta(delta_GN, delta_GD, r)
			#print beta
			delta_final=delta_GD+(beta*subtract(delta_GN, delta_GD))
			print 'Mixed'
		else:
			delta_final=(r/sqrt(sum(square(delta_GD))))*delta_GD
			print 'GD'
	
		
		# Calculate the new H
		new_H=matrix([item for sublist in array(delta_final) for item in sublist]+[0])+array([item for sublist in array(H) for item in sublist])
		new_H=matrix(resize(new_H,(3,3)))
		#print new_H
		temp_F=findF(new_H, best_inlier_set)
		temp_C=sum(square(subtract(X,temp_F)))

		# Calculate rho
		rho=(C-temp_C)/float((2*transpose(delta_final)*transpose(J)*E)-(transpose(delta_final)*transpose(J)*J*delta_final))
		print 'rho', rho 
		
		# Update r based on rho value
		if rho<=0:
			r=r/2
		elif rho<=0.25:
			r=r/4
			H=new_H
		elif rho<=0.75:
			r=r
			H=new_H
		else:
			r=2*r	
			H=new_H
		print 'r', r		
		print H
		
	
		#print 'stop', sqrt(sum(square([item for sublist in array(delta_final) for item in sublist])))
		# If stop conditions are met, stop the optimization
		if ((rho>0)and(rho<0.01)) or (sqrt(sum(square([item for sublist in array(delta_final) for item in sublist]))))<0.05:
			stop=1	
	return H
#--------------------------------------------------------------------------------------------#
# Finds the F vector 
def findF(H, best_inlier_set):
	F=[]
	H=array(H)
	for loopVar1 in range(0, len(best_inlier_set)):
		f1=((H[0][0]*best_inlier_set[loopVar1][0])+(H[0][1]*best_inlier_set[loopVar1][1])+H[0][2])/((H[2][0]*best_inlier_set[loopVar1][0])+(H[2][1]*best_inlier_set[loopVar1][1])+H[2][2])
		f2=((H[1][0]*best_inlier_set[loopVar1][0])+(H[1][1]*best_inlier_set[loopVar1][1])+H[1][2])/((H[2][0]*best_inlier_set[loopVar1][0])+(H[2][1]*best_inlier_set[loopVar1][1])+H[2][2])
		#print f1
		F.append(f1)
		F.append(f2)
	return F
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
# Finds the Jacobian matrix
def findJ(H, best_inlier_set):
	J=[]
	H=array(H)
	for loopVar1 in range(0, len(best_inlier_set)):
		den=((H[2][0]*best_inlier_set[loopVar1][0])+(H[2][1]*best_inlier_set[loopVar1][1])+H[2][2])
		j1=best_inlier_set[loopVar1][0]/den
		j2=best_inlier_set[loopVar1][1]/den
		j3=1/den
		j4=0
		j5=0
		j6=0
		j7=-((H[0][0]*best_inlier_set[loopVar1][0])+(H[0][1]*best_inlier_set[loopVar1][1])+H[0][2])*best_inlier_set[loopVar1][0]/pow(den,2)
		j8=-((H[0][0]*best_inlier_set[loopVar1][0])+(H[0][1]*best_inlier_set[loopVar1][1])+H[0][2])*best_inlier_set[loopVar1][1]/pow(den,2)
		J.append([j1, j2, j3, j4, j5, j6, j7, j8])
		
		j1=0
		j2=0
		j3=0
		j4=best_inlier_set[loopVar1][0]/den
		j5=best_inlier_set[loopVar1][1]/den
		j6=1/den
		j7=-((H[1][0]*best_inlier_set[loopVar1][0])+(H[1][1]*best_inlier_set[loopVar1][1])+H[1][2])*best_inlier_set[loopVar1][0]/pow(den,2)
		j8=-((H[1][0]*best_inlier_set[loopVar1][0])+(H[1][1]*best_inlier_set[loopVar1][1])+H[1][2])*best_inlier_set[loopVar1][1]/pow(den,2)
		J.append([j1, j2, j3, j4, j5, j6, j7, j8])
		
	return matrix(J)
#--------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------#
# Find the beta by solving quadratic equation
def find_beta(delta_GN, delta_GD, r):
	a=sum(square(subtract(delta_GN, delta_GD)))
	b=float(2*transpose(delta_GD)*subtract(delta_GN, delta_GD))
	c=sum(square(delta_GD))-pow(r,2)
	#print a, b, c
	beta=(-b+sqrt(pow(b,2)-4*a*c))/(2*a)
	return beta
#--------------------------------------------------------------------------------------------#

#This function reads the matches saved in a text file.
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

#---------------------------------------------------------------------------------------------#
# This function takes in two images and the matched set of points. Draws lines between matched points on a concatenated image
def draw_lines(matched_img, inliers):
	offset=820
	#print offset
	for loopVar1 in range(0, len(inliers)):		# For each pair of point in the matched set
		#print matches[loopVar1]
		cv2.line(matched_img, (int(inliers[loopVar1][0]), int(inliers[loopVar1][1])), (int(inliers[loopVar1][2])+offset, int(inliers[loopVar1][3])), (0,255,0), 1)		# Draw a line
		cv2.circle(matched_img, (int(inliers[loopVar1][0]), int(inliers[loopVar1][1])), 2, (255,0,0), -1) # Draw a point 
		cv2.circle(matched_img, (int(inliers[loopVar1][2])+offset, int(inliers[loopVar1][3])), 2, (255,0,0), -1) # Draw a point
		
	return matched_img
#---------------------------------------------------------------------------------------------#	
 
matches=readmatches(filename1)	# Read the correspondences between two images
M=(1-e)*len(matches)	# Calculate M
print delta, M, N	
H, best_inlier_set=ransac(matches, delta, M, int(ceil(N))) # Run RANSAC
final_H=dogleg_refinement(H, best_inlier_set)	# Run dog-leg
#final_H=H
final_H=array(final_H) 
print final_H
save_homography(filename2, final_H)	# Save the homography to a file
matched_img=cv2.imread(filename3,1)	# Read the concatenated image
ransac_image=draw_lines(matched_img, best_inlier_set) # Draw lines for inlier set
cv2.imshow('ransac_matched', ransac_image)	# Show the image which shows inliers and outliers
cv2.imwrite(filename4, ransac_image) # Save the image which shows inliers and outliers	
#---------------------------------------------------------------------------------------------#

