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
from scipy.optimize import leastsq
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

#----------------------------------------------------------------------------------------------#
def ExtractParams(K, E):
	p=[]
	p.append(K[0][0])
	p.append(K[0][1])
	p.append(K[1][1])
	p.append(K[0][2])
	p.append(K[1][2])
	for loopVar0 in range(len(E)):
		R=cv.fromarray(copy(matrix(E[loopVar0])[:, 0:3]))
		R_vec=cv.fromarray(zeros((1,3)))
		cv.Rodrigues2(R, R_vec)
		R_vec=asarray(R_vec)
		p.append(R_vec[0][0])
		p.append(R_vec[0][1])
		p.append(R_vec[0][2])
		p.append(E[loopVar0][0][3])
		p.append(E[loopVar0][1][3])
		p.append(E[loopVar0][2][3])
	return p	
#----------------------------------------------------------------------------------------------#
def CombineParams(p):
	K=[[p[0], p[1], p[3]], [0, p[2], p[4]], [0, 0, 1]]
	E=[]
	for loopVar0 in range(0, int((len(p)-5)/6)):
		temp_E=zeros((3,4))
		R=cv.fromarray(zeros((3,3)))
		R_vec=zeros((1,3))
		#print p[(5+(loopVar0*6)):(8+(loopVar0*6))]
		R_vec[0]=copy(p[(5+(loopVar0*6)):(8+(loopVar0*6))])
		R_vec=cv.fromarray(R_vec)
		cv.Rodrigues2(R_vec, R)
		R=asarray(R)
		t=p[(8+(loopVar0*6)):(11+(loopVar0*6))]
		temp_E[:,0]=R[:,0]
		temp_E[:,1]=R[:,1]
		temp_E[:,2]=R[:,2]
		temp_E[:,3]=t
		E.append(array(temp_E).tolist())
	return K, E	
#----------------------------------------------------------------------------------------------#
def CostFunction(p):
	K,E=CombineParams(p)
	K=matrix(K)
	est_x=[]
	for loopVar1 in range(len(E)):
		temp_E=matrix(E[loopVar1])
		P=K*temp_E
		for loopVar2 in range(len(xm)):
			proj_point=P*transpose(matrix(xm[loopVar2]))
			proj_point=asarray(asarray(proj_point/float(proj_point[2])))
			est_x.append([proj_point[0], proj_point[1]])
	est_x=[x for sublist in est_x for x in sublist]
	est_x=[x[0] for x in est_x]
	cost=subtract(X,est_x)
	#print len(est_x), est_x[480], est_x[481], est_x[322], est_x[323]	
	return cost
#----------------------------------------------------------------------------------------------#
	
#Points used for homography calculations - op are in original plane and tp are in transformed plane
#op=matrix([[210, 490],[371, 490],[363, 562],[219, 562]])
#tp=matrix([[116, 56],[428, 56],[428, 271],[116, 271]])

V=[]
y=[]
for loopVar0 in range(1, 41):
	H=matrix(readhomography(filename+str(loopVar0)+'_homography.txt'))
	H=transpose(H)
	H=asarray(H)
	#print H, H[0], H[0][0]
	i=1
	j=2
	V12=[H[i-1][1-1]*H[j-1][1-1], H[i-1][1-1]*H[j-1][2-1]+H[i-1][2-1]*H[j-1][1-1], H[i-1][2-1]*H[j-1][2-1], H[i-1][3-1]*H[j-1][1-1]+H[i-1][1-1]*H[j-1][3-1], H[i-1][3-1]*H[j-1][2-1]+H[i-1][2-1]*H[j-1][3-1], H[i-1][3-1]*H[j-1][3-1]]
	i=1 
	j=1
	V11=[H[i-1][1-1]*H[j-1][1-1], H[i-1][1-1]*H[j-1][2-1]+H[i-1][2-1]*H[j-1][1-1], H[i-1][2-1]*H[j-1][2-1], H[i-1][3-1]*H[j-1][1-1]+H[i-1][1-1]*H[j-1][3-1], H[i-1][3-1]*H[j-1][2-1]+H[i-1][2-1]*H[j-1][3-1], H[i-1][3-1]*H[j-1][3-1]]
	i=2
	j=2
	V22=[H[i-1][1-1]*H[j-1][1-1], H[i-1][1-1]*H[j-1][2-1]+H[i-1][2-1]*H[j-1][1-1], H[i-1][2-1]*H[j-1][2-1], H[i-1][3-1]*H[j-1][1-1]+H[i-1][1-1]*H[j-1][3-1], H[i-1][3-1]*H[j-1][2-1]+H[i-1][2-1]*H[j-1][3-1], H[i-1][3-1]*H[j-1][3-1]]
	
	V.append(V12)
	V.append(subtract(V11,V22))
	y.append(0)
	y.append(0)
#print matrix(V), len(V), y, len(y)
print V[0]
print V[1]
b=linalg.svd(V)[2][5]
print linalg.svd(V)[1]
print 'b=', b
w=[[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]]
print 'w=', matrix(w)
#---------------------------------------------------------------------------------------------#
y0=(w[1-1][2-1]*w[1-1][3-1]-w[1-1][1-1]*w[2-1][3-1])/float(w[1-1][1-1]*w[2-1][2-1]-pow(w[1-1][2-1],2))
lmbd=w[3-1][3-1]-((pow(w[1-1][3-1],2)+y0*(w[1-1][2-1]*w[1-1][3-1]-w[1-1][1-1]*w[2-1][3-1]))/float(w[1-1][1-1]))
alpha_x=sqrt(lmbd/float(w[1-1][1-1]))
alpha_y=sqrt(lmbd*float(w[1-1][1-1])/(w[1-1][1-1]*w[2-1][2-1]-pow(w[1-1][2-1],2)))
s=(-w[1-1][2-1]*pow(alpha_x,2)*alpha_y)/float(lmbd)
x0=(s*y0/float(alpha_y))-(w[1-1][3-1]*pow(alpha_x,2)/float(lmbd))
#print 'x0=', x0
#print 'y0=', y0
#print 'alpha_x', alpha_x
#print 'alpha_y', alpha_y
#print 's', s
K=[[alpha_x, s, x0],[0, alpha_y, y0],[0, 0, 1]]
print 'K=', matrix(K)
save_homography('Intrinsic_Parameters_without_LM.txt', K)

inv_K=inv(matrix(K))
E=[]
X=[]
for loopVar0 in range(1, 41):
	H=matrix(readhomography(filename+str(loopVar0)+'_homography.txt'))
	H=transpose(H)
	H=asarray(H)
	r1=inv_K*transpose(matrix(H[1-1]))
	scaling_factor=norm(r1)
	r1=r1/float(scaling_factor)
	r2=inv_K*transpose(matrix(H[2-1]))/float(scaling_factor)
	r1=asarray(transpose(r1))[0]
	r2=asarray(transpose(r2))[0]
	r3=cross(r1,r2)
	t=inv_K*transpose(matrix(H[3-1]))/float(scaling_factor)
	t=asarray(transpose(t))[0]
	R=[[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]]
	R=matrix(R)
	U,D,V=linalg.svd(R)
	R=asarray(U*V)
	E.append([[R[0][0], R[0][1], R[0][2], t[0]], [R[1][0], R[1][1], R[1][2], t[1]], [R[2][0], R[2][1], R[2][2], t[2]]])
	print '[R|t]=', matrix(E[loopVar0-1]), 'Image:', loopVar0
	save_homography(filename+str(loopVar0)+'_Extrinsic_without_LM.txt', E[loopVar0-1])
	corners=readcorners(filename+str(loopVar0)+'_corners.txt')
	X.append([x for sublist in corners for x in sublist])

#E=[[[1, 0, 0, 20], [0, 1, 0, 30], [0, 0, 1, 40]]]	
X=[x for sublist in X for x in sublist]
xm=readcorners('world_coordinates.txt')
for loopVar1 in range(len(xm)):
	xm[loopVar1]=[xm[loopVar1][0], xm[loopVar1][1], 0, 1]

p=ExtractParams(K, E)
print len(p)
cost=CostFunction(p)
print sum(square(cost))
#print type(cost), len(cost)
#print cost
optimal_p=leastsq(CostFunction, p)[0]	
print optimal_p
K,E=CombineParams(optimal_p)
cost=CostFunction(optimal_p)
print sum(square(cost))
save_homography('Intrinsic_Parameters_LM.txt', K)
for loopVar0 in range(len(E)):
	save_homography(filename+str(loopVar0+1)+'_Extrinsic_LM.txt', E[loopVar0])
