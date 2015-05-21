# Importing libraries
import cv2
import cv
from math import *
from numpy import *
#from sympy import Symbol,cos,sin
from operator import *
from numpy.linalg import *
import time
import ctypes
from scipy.optimize import leastsq
from matplotlib import pyplot as plt
# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

folder='Face Dataset/' # Dataset Folder
NUMBER_OF_SUBJECTS=30	# Number of Subjects in the dataset
NUMBER_OF_IMAGES_PER_SUBJECT=21 # Number of images per subject
MAX_NUMBER_OF_EIGEN_VECTORS=20 # The desired number of maximum eigen vectors that we want to test
#----------------------------------------------------------------------------------------------#
# This function reads all the images in the given folder, converts each image to an array and 
# returns a matrix of image vectors
def readImages(folder):
	imageVectors=[]
	for loopVar1 in range(NUMBER_OF_SUBJECTS):	# For all subjects	
		for loopVar2 in range(NUMBER_OF_IMAGES_PER_SUBJECT): # For all Images for each subject
			img = cv2.imread(folder+str(loopVar1+1).zfill(2)+'_'+str(loopVar2+1).zfill(2)+'.png', 0) # Read the image
			img = asarray(img).flatten().tolist() # Flatten it as a vector
			imageVectors.append(img) # Append the vector to a matrix
	return imageVectors		# Return that matrix

#----------------------------------------------------------------------------------------------#
# This function normalizes the image vectors and then subtracts mean from them.
# Returns the normalized zero-mean vectors
def normalizeVectors(vectors):
	for loopVar1 in range(len(vectors)):	# For each vector
		vectors[loopVar1]=vectors[loopVar1]/norm(vectors[loopVar1]) # Find the normalized vector
	
	meanVector=mean(array(vectors), 0)	# Find the mean-vector
	
	for loopVar1 in range(len(vectors)):	# For each vector
		vectors[loopVar1]=(array(vectors[loopVar1])-meanVector).tolist() # Subtract the mean
	meanVector=mean(array(vectors), 0)	# Just for verification, check if the mean is zero. It should be.
	print meanVector, norm(meanVector)
	return vectors		# Return the normalized zero-mean vectors

#----------------------------------------------------------------------------------------------#
# This function takes in train and test vectors in subspace and classifies a test vector based on
# nearest neighbor classification method. Returns the classified labels for test vectors and accuracy.
def classify(trainFeatures, testFeatures):
	classifiedLabels=[]
	correctClassifications=0			# Counter for correct classifications
	for loopVar1 in range(shape(testFeatures)[1]):  # For each test vector
		testVector=(array(testFeatures[:,loopVar1])).flatten() # Convert it to an array
		querySubject=(loopVar1/NUMBER_OF_IMAGES_PER_SUBJECT)+1 # Find the true label of the test image
		minDistance=1e+10
		for loopVar2 in range(shape(trainFeatures)[1]):	# For train vector
			trainVector=(array(trainFeatures[:,loopVar2])).flatten() # Convert it to an array
			distance=sqrt(sum(square(subtract(trainVector, testVector)))) # Find the euclidean distance
			if distance<minDistance:				# Check if it's the minimum so far
				minDistance=distance		# If yes, save the distance
				matchedSubject=(loopVar2/NUMBER_OF_IMAGES_PER_SUBJECT)+1 # Save the predicted label
		if matchedSubject==querySubject:			# If the predicted label is same as true label
			correctClassifications+=1			# Increase the correct classifications count
		classifiedLabels.append(matchedSubject)			
	accuracy=correctClassifications/float(shape(testFeatures)[1])	# Find the accuracy
	return classifiedLabels, accuracy	# Return the classified labels and the accuracy

#----------------------------------------------------------------------------------------------#
# Main Code starts
trainimageVectors=readImages(folder+'train/')	# Read and vectorize all training images
testimageVectors=readImages(folder+'test/')	# Read and vectorize all test images

trainimageVectors=normalizeVectors(trainimageVectors)   # Get normalized zero-mean vectors for train images
testimageVectors=normalizeVectors(testimageVectors)	# Get normalized zero-mean vectors for test images

print len(trainimageVectors), type(trainimageVectors) 	# Print the length of those vectors for debugging
print len(testimageVectors), type(testimageVectors)

X=matrix(trainimageVectors)		# Make a matrix out of all the training image vectors 
print shape(X)				# X will be (16384 x N)
XXT=X*transpose(X)	# Fing X transpose X (Note the variables name are quite different as initial X in program is (Nx16384))
print shape(XXT)			# X transpose X will be (NxN)
U,D,V=linalg.svd(XXT)			# Find SVD of X transpose X
for num_eig_vec in range(1,MAX_NUMBER_OF_EIGEN_VECTORS+1):	# For eigen vectors from 1 to MAX, find a classifier
	eigenVectors=U[:,0:num_eig_vec]		# Pick top 'p' eigen vectors of X transpose X
	print shape(eigenVectors)		
	W=transpose(X)*eigenVectors		# Find top 'p' eigen vectors of X X transpose by multiplying by X
	print shape(W)				# Size of W will be (16384 x p)
	featureVectors=transpose(W)*transpose(X) # Project the training images onto subspace
	print shape(featureVectors)		# Size of the new train feature vectors will be (px1)

	Xtest=matrix(testimageVectors)				# Make a matrix out of all the test image vectors 
	testfeatureVectors=transpose(W)*transpose(Xtest)	# Project the test images onto subspace
	print shape(testfeatureVectors)		# Size of the new test feature vectors will be (px1)

	classifiedLabels, accuracy=classify(featureVectors, testfeatureVectors) # Classify the test data 
	print 'Accuracy = ', accuracy, 'PCA - Eigen Vectors = ', num_eig_vec	# Print the accuracy
