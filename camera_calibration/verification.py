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

filename='Dataset1/Pic_'
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
		for loopVar2 in range(0, len(H[0])):
			fo.write(str(H[loopVar1][loopVar2]))
			if loopVar2 != len(H[0])-1:
				fo.write('\t')
		if loopVar1 !=len(H)-1:
			fo.write('\n')
	fo.close()	
						
#----------------------------------------------------------------------------------------------#

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
	
def calc_error(orig_points, reproj_points):
	errors=[]
	for loopVar1 in range(len(orig_points)):
		errors.append(sqrt((pow((orig_points[loopVar1][0]-reproj_points[loopVar1][0]),2)+pow((orig_points[loopVar1][1]-reproj_points[loopVar1][1]),2))[0]))
	#print mean(errors), max(errors), std(errors)	
	return mean(errors), max(errors), std(errors)	
#---------------------------------------------------------------------------------------------#
	
image_num=[3, 20, 26, 37]
K_blm=matrix(readhomography('Intrinsic_Parameters_without_LM.txt'))
K_alm=matrix(readhomography('Intrinsic_Parameters_LM.txt'))

E_blm=matrix(readhomography(filename+str(11)+'_Extrinsic_without_LM.txt'))
E_alm=matrix(readhomography(filename+str(11)+'_Extrinsic_LM.txt'))

orig_points=readcorners(filename+str(11)+'_corners.txt')

H_blm_ground=K_blm*delete(E_blm, 2, 1) 
H_alm_ground=K_alm*delete(E_alm, 2, 1) 

for loopVar0 in range(1, 41):#image_num:#range(1, 41):###:
	fixed_img = cv.LoadImage(filename+str(11)+'_just_corners.jpg', 1)
	
	E_blm=matrix(readhomography(filename+str(loopVar0)+'_Extrinsic_without_LM.txt'))
	E_alm=matrix(readhomography(filename+str(loopVar0)+'_Extrinsic_LM.txt'))
	
	H_blm=K_blm*delete(E_blm, 2, 1) 
	inv_H_blm=inv(H_blm)
	H_alm=K_alm*delete(E_alm, 2, 1) 
	inv_H_alm=inv(H_alm)
	
	corners=readcorners(filename+str(loopVar0)+'_corners.txt')
	proj_to_fixed_blm=[]
	proj_to_fixed_alm=[]
	for loopVar1 in range(len(corners)):
		corners[loopVar1]=[corners[loopVar1][0], corners[loopVar1][1], 1]
		
		proj_to_world=inv_H_blm*transpose(matrix(corners[loopVar1]))
		temp_point=asarray(H_blm_ground*proj_to_world)
		temp_point=temp_point/float(temp_point[2])
		proj_to_fixed_blm.append([temp_point[0],temp_point[1]])
		cv.Circle(fixed_img, (int(temp_point[1]), int(temp_point[0])), 2, (0,0,255), -1) # Draw a point
		
		proj_to_world=inv_H_alm*transpose(matrix(corners[loopVar1]))
		temp_point=asarray(H_alm_ground*proj_to_world)
		temp_point=temp_point/float(temp_point[2])
		proj_to_fixed_alm.append([temp_point[0],temp_point[1]])
		cv.Circle(fixed_img, (int(temp_point[1]), int(temp_point[0])), 2, (0,255,0), -1) # Draw a point
		
	#save_corners('Reprojected_'+str(loopVar0)+'_Without_LM.txt', proj_to_fixed_blm)	
	#save_corners('Reprojected_'+str(loopVar0)+'_with_LM.txt', proj_to_fixed_alm)
	cv.SaveImage('Reprojected_'+str(loopVar0)+'.jpg', fixed_img) #Save the result
	
	error_blm=calc_error(orig_points, proj_to_fixed_blm)
	error_alm=calc_error(orig_points, proj_to_fixed_alm)
	print error_blm
	print error_alm
	print loopVar0		
	
'''
corners=readcorners('world_coordinates.txt')	
	
for loopVar0 in image_num:
	img = cv.LoadImage(filename+str(loopVar0)+'.jpg', 1)
	
	E_blm=matrix(readhomography(filename+str(loopVar0)+'_Extrinsic_without_LM.txt'))
	E_alm=matrix(readhomography(filename+str(loopVar0)+'_Extrinsic_LM.txt'))
	P_blm=K_blm*E_blm
	P_alm=K_alm*E_alm
	
	#P_blm_inv=linalg.pinv(P_blm)
	#P_alm_inv=linalg.pinv(P_alm)
	
	proj_blm=[]
	proj_alm=[]
	for loopVar1 in range(len(corners)):
		corners[loopVar1]=[corners[loopVar1][0], corners[loopVar1][1], 0, 1]
		
		#proj_to_world=P_blm_inv*transpose(matrix(corners[loopVar1]))
		temp_point=asarray(P_blm*transpose(matrix(corners[loopVar1])))
		temp_point=temp_point/float(temp_point[2])
		proj_blm.append([temp_point[0],temp_point[1]])
		cv.Circle(img, (int(temp_point[1]), int(temp_point[0])), 2, (0,0,255), -1) # Draw a point
		
		#proj_to_world=P_alm_inv*transpose(matrix(corners[loopVar1]))
		temp_point=asarray(P_alm*transpose(matrix(corners[loopVar1])))
		temp_point=temp_point/float(temp_point[2])
		proj_alm.append([temp_point[0],temp_point[1]])
		cv.Circle(img, (int(temp_point[1]), int(temp_point[0])), 2, (0,255,0), -1) # Draw a point
		
	save_corners('Reprojected_'+str(loopVar0)+'_Without_LM.txt', proj_blm)	
	save_corners('Reprojected_'+str(loopVar0)+'_with_LM.txt', proj_alm)
	cv.SaveImage('Reprojected_'+str(loopVar0)+'.jpg', img) #Save the result		'''
