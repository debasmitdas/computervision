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
filename1='Image3_segmented.jpg'
filename2='Image3_labeled.jpg'
filename3='Image3_letter_'
#----------------------------------------------------------------------------------------------#
def connected_labeling(labeled_img):
	labels=[]
	equivalences=[]
	current_label=0
	for loopVar1 in range(0, labeled_img.rows):		# Loop for all pixels
		for loopVar2 in range(0, labeled_img.cols):	
			if labeled_img[loopVar1, loopVar2]!=0:
				try:
					if (labeled_img[loopVar1-1, loopVar2]!=0)and(labeled_img[loopVar1, loopVar2-1]!=0):
						labeled_img[loopVar1, loopVar2]=labeled_img[loopVar1-1, loopVar2]
						
						if (labeled_img[loopVar1-1, loopVar2]!=labeled_img[loopVar1, loopVar2-1]):
							#print labeled_img[loopVar1-1, loopVar2], labeled_img[loopVar1, loopVar2-1]
							index_1=index_2d(equivalences,labeled_img[loopVar1-1, loopVar2])
							index_2=index_2d(equivalences,labeled_img[loopVar1, loopVar2-1])
							
							if (index_1==None and index_2!=None):
								equivalences[index_2[0]].append(labeled_img[loopVar1-1, loopVar2])
								#print "append"
							elif (index_1!=None and index_2==None):
								equivalences[index_1[0]].append(labeled_img[loopVar1, loopVar2-1])
								#print "append"
							elif (index_1!=None and index_2!=None):
								if index_1[0]!=index_2[0]:
									for item in equivalences[index_2[0]]:
										equivalences[index_1[0]].append(item)
									equivalences.pop(index_2[0])
									#print "merge"		
							else:
								equivalences.append([labeled_img[loopVar1-1, loopVar2],labeled_img[loopVar1, loopVar2-1]])					
								#print "create"
					elif labeled_img[loopVar1-1, loopVar2]!=0:
						labeled_img[loopVar1, loopVar2]=labeled_img[loopVar1-1, loopVar2]
					
					elif labeled_img[loopVar1, loopVar2-1]!=0:
						labeled_img[loopVar1, loopVar2]=labeled_img[loopVar1, loopVar2-1]
							
					else:
						current_label+=1
						labels.append(current_label)
						equivalences.append([current_label])
						labeled_img[loopVar1, loopVar2]=current_label			
				except IndexError:
					labeled_img[loopVar1, loopVar2]=0
					
		print loopVar1, "labeling"
	#print "equivalences", equivalences, len(equivalences)
	labels=[]
	for loopVar1 in range(0, labeled_img.rows):		# Loop for all pixels
		for loopVar2 in range(0, labeled_img.cols):	
			if labeled_img[loopVar1, loopVar2]!=0:
				index=index_2d(equivalences,labeled_img[loopVar1, loopVar2])
				label=min(equivalences[index[0]])
				labeled_img[loopVar1, loopVar2]=label
				if not(label in labels):
					labels.append(label)
		print loopVar1, "equivalence removal"
	return labeled_img, labels			
			
#---------------------------------------------------------------------------------------------#	
def extract_letters(labeled_img, labels):
	images=[]
	image_boundaries=[]
	for item in labels:
		images.append(cv.CreateMat(labeled_img.rows, labeled_img.cols, cv.CV_8UC1))
		image_boundaries.append([1e+10, 0, 1e+10, 0]) # x_min, x_max, y_min, y_max
	for loopVar1 in range(0, labeled_img.rows):		# Loop for all pixels
		for loopVar2 in range(0, labeled_img.cols):
			if labeled_img[loopVar1, loopVar2]!=0:
				image_index=labels.index(labeled_img[loopVar1, loopVar2])	
				images[image_index][loopVar1, loopVar2]=255
				if loopVar1<image_boundaries[image_index][0]:
					image_boundaries[image_index][0]=loopVar1
				if loopVar1>image_boundaries[image_index][1]:
					image_boundaries[image_index][1]=loopVar1
				if loopVar2<image_boundaries[image_index][2]:
					image_boundaries[image_index][2]=loopVar2
				if loopVar2>image_boundaries[image_index][3]:
					image_boundaries[image_index][3]=loopVar2		
				labeled_img[loopVar1, loopVar2]=int(20+(image_index*(255-20)/26))
		print loopVar1, "Letter Extraction"
	for loopVar1 in range(len(images)):
		images[loopVar1]=asarray(images[loopVar1][:,:])
		images[loopVar1]=images[loopVar1][image_boundaries[loopVar1][0]-5:image_boundaries[loopVar1][1]+5, image_boundaries[loopVar1][2]-5:image_boundaries[loopVar1][3]+5]		 
	print image_boundaries					
	return images, labeled_img			
#---------------------------------------------------------------------------------------------#	
#---------------------------------------------------------------------------------------------#	
def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))
#---------------------------------------------------------------------------------------------#	       
# Main code
# Load input image

orig_img = cv2.imread(filename1,0)
th, binary_img=cv2.threshold(orig_img, 127, 255, cv2.THRESH_BINARY)
binary_img=cv.fromarray(binary_img)
input_img=cv.CreateMat(orig_img.shape[0], orig_img.shape[1], cv.CV_16UC1)
print type(input_img)
cv.Convert(binary_img, input_img)
labeled_img, labels=connected_labeling(input_img)

extracted_letter_imgs=[]	
extracted_letter_imgs, labeled_img = extract_letters(labeled_img, labels)

letter_count=0
labels=[]
for loopVar1 in range(len(extracted_letter_imgs)):
	if not(extracted_letter_imgs[loopVar1].shape[0]<50 and extracted_letter_imgs[loopVar1].shape[1]<50):
		cv2.imwrite(filename3+str(letter_count+1)+".jpg", extracted_letter_imgs[loopVar1]) #Save the result
		letter_count+=1
		labels.append(letter_count)
cv.SaveImage(filename2, labeled_img) #Save the result

print "labels", labels, len(labels)
cv2.waitKey(0) #Wait for key-press'''
