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
NUMBER_OF_SUBJECTS=30  # Number of Subjects in the dataset
NUMBER_OF_IMAGES_PER_SUBJECT=21 # Number of images per subject
MAX_NUMBER_OF_EIGEN_VECTORS=20 # The desired number of maximum eigen vectors that we want to test

#----------------------------------------------------------------------------------------------#
# This function reads all the images in the given folder, converts each image to an array and 
# returns a matrix of image vectors
def readImages(folder):
	imageVectors=[]
	for loopVar1 in range(NUMBER_OF_SUBJECTS):
		for loopVar2 in range(NUMBER_OF_IMAGES_PER_SUBJECT):
			img = cv2.imread(folder+str(loopVar1+1).zfill(2)+'_'+str(loopVar2+1).zfill(2)+'.png', 0) # Read two images
			img = asarray(img).flatten().tolist()
			imageVectors.append(img)
	return imageVectors

#----------------------------------------------------------------------------------------------#
# This function normalizes the image vectors and then subtracts mean from them.
# Returns the normalized zero-mean vectors
def normalizeVectors(vectors):
	for loopVar1 in range(len(vectors)):	# For each vector
		vectors[loopVar1]=vectors[loopVar1]/norm(vectors[loopVar1])  # Find the normalized vector
	
	meanVector=mean(array(vectors), 0)	# Find the mean-vector
	
	for loopVar1 in range(len(vectors)):	# For each vector
		vectors[loopVar1]=(array(vectors[loopVar1])-meanVector).tolist()  # Subtract the mean
	meanVector=mean(array(vectors), 0)	# Just for verification, check if the mean is zero. It should be.
	print meanVector, norm(meanVector)	
	return vectors			# Return the normalized zero-mean vectors

#----------------------------------------------------------------------------------------------#
# This function takes in a full set of vectors and returns the means of each class within the input vectors
def findClassMean(vectors):
	classmeanMatrix=[]
	vectors=matrix(array(vectors))	# Convert the received vectors to matrix
	for loopVar1 in range(NUMBER_OF_SUBJECTS):	# For each class (subject)
		classmeanMatrix.append(mean(array(vectors[(loopVar1*NUMBER_OF_IMAGES_PER_SUBJECT):((loopVar1+1)*NUMBER_OF_IMAGES_PER_SUBJECT),:]),0))	# Find the mean and append it to class mean array
	classmeanMatrix=matrix(array(classmeanMatrix))	# Convert the class mean array to a matrix
	return classmeanMatrix		# Return that matrix

#----------------------------------------------------------------------------------------------#
# This function takes in Z and the training vectors. Computes Z transpose S Z.
def computeZTSwZ(Z, vectors):
	ZT=transpose(Z)				# Find transpose of Z
	ZTSwZ=matrix(zeros((NUMBER_OF_SUBJECTS, NUMBER_OF_SUBJECTS)))	# Create a matrix for Z transpose S Z
	for loopVar1 in range(NUMBER_OF_SUBJECTS):		# For each class (subject)
		temp_matrix=matrix(zeros((NUMBER_OF_SUBJECTS, NUMBER_OF_SUBJECTS)))	# Create a temporary matrix
		for loopVar2 in range(NUMBER_OF_IMAGES_PER_SUBJECT):	# For each image in each subject
			vector=matrix(array(vectors[(loopVar1*NUMBER_OF_IMAGES_PER_SUBJECT)+loopVar2])) # Get its vector
			temp_matrix+=(ZT*transpose(vector)*vector*Z)	# Find (Z transpose x) times (x transpose Z)
		temp_matrix=temp_matrix/NUMBER_OF_IMAGES_PER_SUBJECT	# Normalize the temporary matrix by class size
		ZTSwZ+=temp_matrix				# Update Z transpose S Z
	ZTSwZ=ZTSwZ/len(vectors)				# Normalize the Z transpose S Z matrix by number of classes
	return ZTSwZ				# Return Z transpose S Z

#----------------------------------------------------------------------------------------------#
# This function takes in train and test vectors in subspace and classifies a test vector based on
# nearest neighbor classification method. Returns the classified labels for test vectors and accuracy.
def classify(trainFeatures, testFeatures):
	classifiedLabels=[]	
	correctClassifications=0			# Counter for correct classifications
	for loopVar1 in range(shape(testFeatures)[1]):	# For each test vector
		testVector=(array(testFeatures[:,loopVar1])).flatten()	# Convert it to an array
		querySubject=(loopVar1/NUMBER_OF_IMAGES_PER_SUBJECT)+1  # Find the true label of the test image
		minDistance=1e+10
		for loopVar2 in range(shape(trainFeatures)[1]): # For train vector
			trainVector=(array(trainFeatures[:,loopVar2])).flatten() # Convert it to an array
			distance=sqrt(sum(square(subtract(trainVector, testVector)))) # Find the euclidean distance
			if distance<minDistance:			# Check if it's the minimum so far
				minDistance=distance			# If yes, save the distance
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

print len(trainimageVectors), type(trainimageVectors)	# Print the length of those vectors for debugging
print len(testimageVectors), type(testimageVectors)

X=matrix(trainimageVectors)			# Make a matrix out of all the training image vectors 
M=findClassMean(trainimageVectors)		# Find the class mean matrix, M
print shape(M)					# Size of M will be (16384 x K)
MMT=(M*transpose(M))/NUMBER_OF_SUBJECTS		# Find M transpose M
print shape(MMT)				# Size of M transpose M will be (K x K)
U,D,V=linalg.svd(MMT)				# Find SVD of M transpose M
Y=transpose(M)*U				# Find eigen vectors of M M transpose
print shape(Y)					# Size of Y (eigen vector matrix) will be (16384 x K)
Diag=matrix(zeros((len(D),len(D))))		# Find Diagonal matrix D using a for-loop
for loopVar1 in range(len(D)):
	Diag[loopVar1, loopVar1]=D[loopVar1]
D=Diag						
print shape(D)					# D is of size (K x K)
Z=Y*inv(D)					# Find Z
print shape(Z)					# Z is of size (16384 x K)
ZTSwZ=computeZTSwZ(Z, trainimageVectors)	# Compute Z transpose S Z
print shape(ZTSwZ)				# Z transpose S Z is of size (K x K)
U,D,V=linalg.svd(ZTSwZ)				# Find SVD of Z transpose S Z
for loopVar1 in range(shape(U)[1]):		# Sort the columns in U so that eigen-vector with lowest eigen value is first
	temp_column=U[:,loopVar1]
	U[:,loopVar1]=U[:,shape(U)[1]-1-loopVar1]
	U[:,shape(U)[1]-1-loopVar1]=temp_column
print shape(U)					# U is of size (K x K)

for num_eig_vec in range(1, MAX_NUMBER_OF_EIGEN_VECTORS+1):	# For eigen vectors from 1 to MAX, find a classifier
	eigenVectors=U[:,0:num_eig_vec]		# Pick first 'p' eigen vectors of Z transpose S Z
	print shape(eigenVectors)
	W=Z*eigenVectors			# Find first 'p' eigen vectors which forms the subspace
	print shape(W)				# Size of W will be (16384 x p)
	featureVectors=transpose(W)*transpose(X)	# Project the training images onto subspace
	print shape(featureVectors)			# Size of the new train feature vectors will be (px1)

	Xtest=matrix(testimageVectors)			# Make a matrix out of all the test image vectors 
	testfeatureVectors=transpose(W)*transpose(Xtest)	# Project the test images onto subspace
	print shape(testfeatureVectors)			# Size of the new test feature vectors will be (px1)

	classifiedLabels, accuracy=classify(featureVectors, testfeatureVectors)	# Classify the test data 
	print 'Accuracy = ', accuracy, 'LDA - Eigen Vectors = ', num_eig_vec	# Print the accuracy
