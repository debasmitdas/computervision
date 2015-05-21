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

folder='Car Dataset/' 			# Dataset folder
NUMBER_OF_TRAIN_POS_IMAGES=710		# Number of postive training images
NUMBER_OF_TRAIN_NEG_IMAGES=1758		# Number of negative training images
TOTAL_NUMBER_OF_TRAIN_IMAGES=NUMBER_OF_TRAIN_POS_IMAGES+NUMBER_OF_TRAIN_NEG_IMAGES	# Total training images

NUMBER_OF_TEST_POS_IMAGES=178		# Number of postive test images
NUMBER_OF_TEST_NEG_IMAGES=440		# Number of negative test images
TOTAL_NUMBER_OF_TEST_IMAGES=NUMBER_OF_TEST_POS_IMAGES+NUMBER_OF_TEST_NEG_IMAGES		# Total test images

TEST_POS_IMG_NUMBERING_OFFSET=710	# Numbering offset while image reading 
TEST_NEG_IMG_NUMBERING_OFFSET=1758

REQUIRED_FP_RATE=0.0001			# Desired FP rate for overall classifier
FP_RATE_FOR_EACH_STAGE=0.5		# Desired FP rate for each stage

#----------------------------------------------------------------------------------------------#
# This function reads in each image and calls another function to compute its HAAR features.
# Stores all the HAAR features of all the images in a matrix.
def readImageHaar(folder, NUMBER_OF_POS_IMAGES, NUMBER_OF_NEG_IMAGES, OFFSET):
	posfolder=folder+'positive/'			# Folder to read positive images from
	for loopVar1 in range(NUMBER_OF_POS_IMAGES):	# For each positive image
		img = cv2.imread(posfolder+str(loopVar1+1+OFFSET).zfill(6)+'.png', 0) # Read the image
		scalingFactor=norm(img.flatten())		# Normalize the image
		if scalingFactor!=0:
			img=img/scalingFactor
		integralImg=zeros((img.shape[0]+1, img.shape[1]+1))	# Find the integral image		
		integralImg[1:,1:]=cumsum(cumsum(img, axis=0, dtype=float64), axis=1, dtype=float64)
		features=calcHaarFeatures(integralImg)		# Calculate HAAR features using the integral image
		features.append(1)				# Append the true label along with HAAR features array
		sampleWeights[loopVar1, 0]=1/float(2*NUMBER_OF_POS_IMAGES)	# Assign uniform weight for this image 
		haarFeatures[loopVar1,:]=array(features)		# Store the HAAR features along with label as a row in the matrix
		print 'Reading Pos Images', loopVar1	# printing the index of the image, for debugging
		
	negfolder=folder+'negative/'		# Folder to read negative images from
	for loopVar1 in range(NUMBER_OF_NEG_IMAGES):	# For each negative image
		img = cv2.imread(negfolder+str(loopVar1+1+OFFSET).zfill(6)+'.png', 0) # Read the image
		scalingFactor=norm(img.flatten())		# Normalize the image
		if scalingFactor!=0:
			img=img/scalingFactor
		integralImg=zeros((img.shape[0]+1, img.shape[1]+1))		# Find the integral image	
		integralImg[1:,1:]=cumsum(cumsum(img, axis=0, dtype=float64), axis=1, dtype=float64)
		features=calcHaarFeatures(integralImg)		# Calculate HAAR features using the integral image
		features.append(0)			# Append the true label along with HAAR features array
		sampleWeights[loopVar1+NUMBER_OF_POS_IMAGES, 0]=1/float(2*NUMBER_OF_NEG_IMAGES)	# Assign uniform weight for this image 
		haarFeatures[loopVar1+NUMBER_OF_POS_IMAGES, :]=array(features) # Store the HAAR features along with label as a row in the matrix
		print 'Reading Neg Images', loopVar1	# printing the index of the image, for debugging
	
	return 0			# Returns nothing as it works with a global HAAR feature Matrix

#----------------------------------------------------------------------------------------------#
# This function takes in an integral image and calculates 166000 haar features and returns them as a list
def calcHaarFeatures(integralImg):
	features=[]
	for loopVar1 in range(1, integralImg.shape[0]):			# For each type of horizontal HAAR window
		for loopVar2 in range(2, integralImg.shape[1], 2):
		
			for loopVar3 in range(0, integralImg.shape[0]-1):	# For each position of the window in the image
				for loopVar4 in range(0, integralImg.shape[1]-1):
					if ((loopVar4+(loopVar2/2))<integralImg.shape[1])and((loopVar4+loopVar2)<integralImg.shape[1])and((loopVar3+loopVar1)<integralImg.shape[0]):					# Check if the window is within the image boundaries
						A=[loopVar3, loopVar4]			# Get all 6 corners of the HAAR rectangle
						B=[loopVar3, loopVar4+(loopVar2/2)]
						C=[loopVar3, loopVar4+loopVar2]
						D=[loopVar3+loopVar1, loopVar4+loopVar2]
						E=[loopVar3+loopVar1, loopVar4+(loopVar2/2)]
						F=[loopVar3+loopVar1, loopVar4]
						feature=-integralImg[A[0], A[1]]+(2*integralImg[B[0], B[1]])-integralImg[C[0], C[1]]+integralImg[D[0], D[1]]-(2*integralImg[E[0], E[1]])+integralImg[F[0], F[1]]	# Calculate the HAAR feature using 6 summations
						features.append(feature)	# Append the feature to the entire list of features

	for loopVar1 in range(2, integralImg.shape[0], 2):		# For each type of vertical HAAR window
		for loopVar2 in range(1, integralImg.shape[1]):

			for loopVar3 in range(0, integralImg.shape[0]-1):	# For each position of the window in the image
				for loopVar4 in range(0, integralImg.shape[1]-1):
					if ((loopVar4+loopVar2)<integralImg.shape[1])and((loopVar3+(loopVar1/2))<integralImg.shape[0])and((loopVar3+loopVar1)<integralImg.shape[0]):					# Check if the window is within the image boundaries
						A=[loopVar3, loopVar4]			# Get all 6 corners of the HAAR rectangle
						B=[loopVar3, loopVar4+loopVar2]
						C=[loopVar3+(loopVar1/2), loopVar4+loopVar2]
						D=[loopVar3+loopVar1, loopVar4+loopVar2]
						E=[loopVar3+loopVar1, loopVar4]
						F=[loopVar3+(loopVar1/2), loopVar4]
						feature=-(-integralImg[A[0], A[1]]+integralImg[B[0], B[1]]-(2*integralImg[C[0], C[1]])+integralImg[D[0], D[1]]-integralImg[E[0], E[1]]+(2*integralImg[F[0], F[1]]))	# Calculate the HAAR feature using 6 summations
						features.append(feature)	# Append the feature to the entire list of features
	
	return features			# Return the computed list of featues for an image

#----------------------------------------------------------------------------------------------#
# This function learns a weak classifier based on the Haar Feature Matrix and corresponding weights (both global variables)
# Returns the weak classifier's feature index, threshold, polarity, trust and beta
def learnWeakClassifier():
	sampleWeights[:, :]=sampleWeights[:, :]/sum(sampleWeights[:, :])	# Normalize the weights

	TPlus=0
	TMinus=0
	for loopVar1 in range(haarFeatures.shape[0]):			# This loop finds TPlus and TMinus values
		if haarFeatures[loopVar1, haarFeatures.shape[1]-1]==1:
			TPlus+=sampleWeights[loopVar1, 0]
		else:
			TMinus+=sampleWeights[loopVar1, 0]

	globalErrors=[]
	globalPolarities=[]
	globalThresholds=[]				# This loop finds the best feature-threshold pair
	for loopVar1 in range(haarFeatures.shape[1]-1):	# For each HAAR feature
		subMatrix=hstack((haarFeatures[:, loopVar1:(loopVar1+1)], haarFeatures[:, haarFeatures.shape[1]-1:haarFeatures.shape[1]], sampleWeights[:, :]))						# Extract the feature column, its labels and weights
		subMatrix=matrix(sorted(array(subMatrix), key=itemgetter(0)))	# Sort the sub-matrix based on feature values
		SPlus=TPlus						# Splus starts from TPlus
		SMinus=TMinus						# SMinus starts from TMinus
		errors=[]
		polarities=[]

		for loopVar2 in range(subMatrix.shape[0]):	# This loop calculates SPlus and SMinus for each possible threshold
			if subMatrix[loopVar2, subMatrix.shape[1]-2]==1:	# If true label is 1, SPlus is decremented
				SPlus-=subMatrix[loopVar2, subMatrix.shape[1]-1]
			else:							# If true label is 0, SMinus is decremented
				SMinus-=subMatrix[loopVar2, subMatrix.shape[1]-1]
			if (SPlus+TMinus-SMinus)<(SMinus+TPlus-SPlus):		# Find the polarity and the error
				errors.append(SPlus+TMinus-SMinus)
				polarities.append(1)
			else:					
				errors.append(SMinus+TPlus-SPlus)
				polarities.append(-1)

		minerror=min(errors)			# Find the minimum error among all errors for each threshold
		globalErrors.append(minerror)		
		globalPolarities.append(polarities[errors.index(minerror)])	# Find corresponding polarity
		globalThresholds.append(subMatrix[errors.index(minerror), 0])	# Find corresponding threshold

	finalError=min(globalErrors)		# Find the minimum error among all errors for each feature
	featureIndex=globalErrors.index(finalError)	# Find the best feature index
	featurePolarity=globalPolarities[globalErrors.index(finalError)] # Find the corresponding polarity
	featureThreshold=globalThresholds[globalErrors.index(finalError)] # Find the corresponding threshold
	# At this point, we are done finding the weak classifer

	beta=finalError/float(1-finalError)		# Find the beta value for the weak classifier
	print 'beta=', beta
	if beta==0 or beta<0:	# If beta is zero or less than zero (because of floating point issues), assign high trust value
		featureTrust=1e+8
	else:		
		featureTrust=log(1/beta)	# Else, find the actual trust value
	
	# Now, use the weak classifier to classify the training images and increase weights of misclassified images
	for loopVar1 in range(haarFeatures.shape[0]):		# For each image 
		if (featurePolarity*haarFeatures[loopVar1, featureIndex]) <= (featurePolarity*featureThreshold): 
			if haarFeatures[loopVar1, haarFeatures.shape[1]-1]==1: # If predicted and true labels are same
				weightMultiple=beta			# Multiplying factor will be 'beta'
			else:					# If predicted and true labels are not same
				weightMultiple=1			# Weights doesn't decrease
		else:
			if haarFeatures[loopVar1, haarFeatures.shape[1]-1]==0:	# If predicted and true labels are same
				weightMultiple=beta			# Multiplying factor will be 'beta'
			else:					# If predicted and true labels are not same
				weightMultiple=1			# Weights doesn't decrease
		sampleWeights[loopVar1, 0]=sampleWeights[loopVar1, 0]*weightMultiple	# Update the weights for next iteration
	
	return featureIndex, featurePolarity, featureThreshold, featureTrust, beta	# Return the weak classifier information

#----------------------------------------------------------------------------------------------#
# This function learns a strong classifier based on the Haar Feature Matrix and corresponding weights (both global variables)
# This function calls 'learnWeakClassifier' function multiple times until the desired FP rate for each stage is achieved
# Returns the information about the strong classifier learned
def learnStrongClassifier():
	global haarFeatures, sampleWeights		
	FP=1.0				# Before learning any weak classifier, the FP for this stage will be 1.0
	weakClassifierIndices=[]
	weakClassifierPolarities=[]
	weakClassifierThresholds=[]
	weakClassifierTrusts=[]

	while (FP >= FP_RATE_FOR_EACH_STAGE):	# Learn weak classifier until desired FP rate for this stage is achieved
		print '******************Learning Weak Classifier*******************'
		featureIndex, featurePolarity, featureThreshold, featureTrust, beta=learnWeakClassifier()	# Learn a weak classifier
		weakClassifierIndices.append(featureIndex)		# Store the new weak classifier's information
		weakClassifierPolarities.append(featurePolarity)
		weakClassifierThresholds.append(featureThreshold)
		weakClassifierTrusts.append(featureTrust)
		print 'Weak Classifier ', len(weakClassifierIndices)
		print featureIndex, featurePolarity, featureThreshold, featureTrust
	
		# We use the set of weak classifiers learned to find strong classifier threshold and FP rate
		weightedDecisionsforPositives=[]
		TotalPositives=0				# This loop finds the strong classifier threshold so that TP=1.0
		for loopVar1 in range(haarFeatures.shape[0]):	# For each image
			if haarFeatures[loopVar1, haarFeatures.shape[1]-1]==1:	# If true label is 1 (positive image)
				weightedDecision=0
				TotalPositives+=1				# Increment the number of total positives
				for loopVar0 in range(len(weakClassifierIndices)):	# For each weak classifier
					if (weakClassifierPolarities[loopVar0]*haarFeatures[loopVar1, weakClassifierIndices[loopVar0]]) <= (weakClassifierPolarities[loopVar0]*weakClassifierThresholds[loopVar0]):	# Find weak classifier's decision
						weightedDecision+=weakClassifierTrusts[loopVar0]*1 # Find weighted summation of such decisions
				weightedDecisionsforPositives.append(weightedDecision)	# Store the decision of strong classifier for all positive images
		strongClassifierThreshold=min(weightedDecisionsforPositives)	# Strong classifier threshold will be minimum among all weighted decisions. This makes sure TP for each strong classifier is 1.0
		
		print 'Strong Classifier Threshold', strongClassifierThreshold
		
		falsePositives=0
		TotalNegatives=0
		TrueNegativeIndices=[]			# This loop finds the number false positives and the true negatives
		for loopVar1 in range(haarFeatures.shape[0]):	# For each image
			if haarFeatures[loopVar1, haarFeatures.shape[1]-1]==0:	# If true label is 0 (negative image)
				TotalNegatives+=1
				weightedDecision=0				
				for loopVar0 in range(len(weakClassifierIndices)):	# For each weak classifier
					if (weakClassifierPolarities[loopVar0]*haarFeatures[loopVar1, weakClassifierIndices[loopVar0]]) <= (weakClassifierPolarities[loopVar0]*weakClassifierThresholds[loopVar0]):		# Find weak classifier's decision
						weightedDecision+=weakClassifierTrusts[loopVar0]*1 # Find weighted summation of such decisions
				
				if weightedDecision >= strongClassifierThreshold: # If summation is greater than threshold
					falsePositives+=1			# Declare it as a false positive
				else:						# Otherwise
					TrueNegativeIndices.append(loopVar1)	# Save it as true negative in order to discard it later
		FP=falsePositives/float(TotalNegatives)		# Find FP rate for this stage
		print 'FP=', FP
		if beta==0 or beta=='nan': # If the latest weak classifier had zero error, then we proceed with next stage, so that the weights which are currently all zeros (because of beta) will be re-initialized in the next stage
			break
		
	#-----------------------------------------------------------------#
	# This part of code removes the haar features of True Negatives from the matrix in-place so that the matrix is not duplicated and memory shortage issues doesn't occur. Idea is to move all the unwanted rows to the end of the matrix and resize the matrix.
	IndicestoRemove=[x for x in TrueNegativeIndices if x<(haarFeatures.shape[0]-len(TrueNegativeIndices))] # Find rows to remove
	IndicestoReplace=[]

	# This loop finds the rows at the end of the matrix which can be used for replacement
	IndextoReplace=(haarFeatures.shape[0]-len(TrueNegativeIndices))	
	for loopVar1 in range(len(IndicestoRemove)):	# For each row to be removed
		replaceFound=0
		while(replaceFound==0):			# Find a replacement row at the end of the matrix
			if not(IndextoReplace in TrueNegativeIndices):
				IndicestoReplace.append(IndextoReplace)	# Remember that replacement row
				replaceFound=1
			IndextoReplace+=1

	# This loop exchanges rows to be removed with rows at the end of the matrix
	for loopVar1 in range(len(IndicestoRemove)):		# For each row to be removed		
		rowtoKeep=array(haarFeatures[IndicestoReplace[loopVar1], :])	# Get the row to be kept but it is at the end of the matrix
		rowtoDelete=array(haarFeatures[IndicestoRemove[loopVar1], :])	# Get the row to be removed
		haarFeatures[IndicestoReplace[loopVar1], :]=rowtoDelete	# Put the row to be removed at the other row's place
		haarFeatures[IndicestoRemove[loopVar1], :]=rowtoKeep	# Put the row to be kept at the removed row's place
		
		tempValue=sampleWeights[IndicestoReplace[loopVar1], 0]		# Similarly, exchange the weights array as well
		sampleWeights[IndicestoReplace[loopVar1], 0]=sampleWeights[IndicestoRemove[loopVar1], 0]
		sampleWeights[IndicestoRemove[loopVar1], 0]=tempValue
	
	# At this point all the rows to be removed are at the end of the matrix and other retained rows are swapped.
	# We simply resize the HAAR feature matrix and corresponding weights array so that the reduced feature set is used by next stage
	haarFeatures.resize((haarFeatures.shape[0]-len(TrueNegativeIndices), 166001), refcheck=False )
	sampleWeights.resize((sampleWeights.shape[0]-len(TrueNegativeIndices), 1), refcheck=False)
	#-----------------------------------------------------------------#
	
	# This loop re-initializes the weights based on Total positives and False positives in the reduced feature 
	for loopVar1 in range(haarFeatures.shape[0]):	# For each image
		if haarFeatures[loopVar1, haarFeatures.shape[1]-1]==1:
			sampleWeights[loopVar1, 0]=1/float(2*TotalPositives)	# Re-initialize weight
		else:
			sampleWeights[loopVar1, 0]=1/float(2*falsePositives)	# Re-initialize weight
	
	return [weakClassifierIndices, weakClassifierPolarities, weakClassifierThresholds, weakClassifierTrusts, FP, strongClassifierThreshold]		# Return all information of this learned strong classifier

#----------------------------------------------------------------------------------------------#
# This function learns a cascaded adaboost classifier by simply calling 'learnStrongClassifier' 
# until the desired global FP rate is achieved. Returns all the strong classifiers learned.
def learnCascadeClassifier():
	strongClassifiers=[]
	globalFalsePositiveRate=1.0		# The global false positive rate will be 1.0 initially
	while (globalFalsePositiveRate > REQUIRED_FP_RATE):	# Repeat until desired overall FP rate is achieved
		print '***********************Learning Strong Classifier***************************'
		strongClassifer=learnStrongClassifier()		# Learn a strong classifier (one stage)
		globalFalsePositiveRate*=strongClassifer[4]	# Update global false positive rate
		
		print 'GFPR=', globalFalsePositiveRate	
		strongClassifiers.append(strongClassifer)	# Store the learned strong classifier's information
		print strongClassifer, len(strongClassifiers)	# Print the number of strong classifiers learned so far

	return strongClassifiers		# Return all the information about all strong classifiers (stages) learned

#----------------------------------------------------------------------------------------------#
# This function takes in a learned cascaded adaboost classifier and the test dataset and classifies the test images
# Returns the Final Accuracy, TP, FN, FP, TN and stage-wise TP, FN, FP, TN values.
def classifyTestImages(finalCascadeClassifier, folder, NUMBER_OF_POS_IMAGES, NUMBER_OF_NEG_IMAGES, OFFSET_POS, OFFSET_NEG):
	classifiedLabels=[]			# Initialize few variables
	correctClassifications=0
	TruePositives=0 
	FalseNegatives=0
	FalsePositives=0
	TrueNegatives=0
	eachStageDecisions=[]

	posfolder=folder+'positive/'		# Folder for positive test images
	print '*********************Classifying Positive Test Images*************************'
	for loopVar1 in range(NUMBER_OF_POS_IMAGES):	# For each positive image
		img = cv2.imread(posfolder+str(loopVar1+1+OFFSET_POS).zfill(6)+'.png', 0) # Read the image
		scalingFactor=norm(img.flatten())		# Normalize the image
		if scalingFactor!=0:
			img=img/scalingFactor
		integralImg=zeros((img.shape[0]+1, img.shape[1]+1))	# Find the integral image	
		integralImg[1:,1:]=cumsum(cumsum(img, axis=0, dtype=float64), axis=1, dtype=float64)
		features=calcHaarFeatures(integralImg)	# Calculate HAAR features
		
		stageLevelDecisions=[]
		
		# These nested loops apply cascaded adaboost to the test image and classifies them
		for loopVar2 in range(len(finalCascadeClassifier)):	# For each stage
			strongClassifier=finalCascadeClassifier[loopVar2]	# Get the strong classifier's info
			weightedDecision=0
			for loopVar3 in range(len(strongClassifier[0])):	# For each weak classifier
				if (strongClassifier[1][loopVar3]*features[strongClassifier[0][loopVar3]]) <= (strongClassifier[1][loopVar3]*strongClassifier[2][loopVar3]):				# Find the decision
						weightedDecision+=strongClassifier[3][loopVar3]*1	# Find weighted decision
				
			if weightedDecision >= strongClassifier[5]:# If weighted decision is greater than strong classifier's threshold
				stageLevelDecisions.append(1)		# Declare the image to be Positive at this stage
			else:						# Else, 
				stageLevelDecisions.append(0)		# Declare the image to be negative at this stage

		eachStageDecisions.append(stageLevelDecisions)		# Save all the stagelevel decisions

		if all(stageLevelDecisions):		# If all stages said, "Positive", declare positive
			TruePositives+=1	
			classifiedLabels.append(1)
			correctClassifications+=1	
		else:					# Otherwise, declare negative
			FalseNegatives+=1	
			classifiedLabels.append(0)

			
	negfolder=folder+'negative/'		# Folder for negative test images
	print '*********************Classifying Negative Test Images*************************'
	for loopVar1 in range(NUMBER_OF_NEG_IMAGES):	# For each negative image
		img = cv2.imread(negfolder+str(loopVar1+1+OFFSET_NEG).zfill(6)+'.png', 0) # Read the image
		scalingFactor=norm(img.flatten())		# Normalize the image
		if scalingFactor!=0:
			img=img/scalingFactor
		integralImg=zeros((img.shape[0]+1, img.shape[1]+1))	# Find the integral image		
		integralImg[1:,1:]=cumsum(cumsum(img, axis=0, dtype=float64), axis=1, dtype=float64)
		features=calcHaarFeatures(integralImg)	# Calculate HAAR features
		
		# These nested loops apply cascaded adaboost to the test image and classifies them
		stageLevelDecisions=[]
		for loopVar2 in range(len(finalCascadeClassifier)):	# For each stage
			strongClassifier=finalCascadeClassifier[loopVar2]	# Get the strong classifier's info
			weightedDecision=0
			for loopVar3 in range(len(strongClassifier[0])):	# For each weak classifier
				if (strongClassifier[1][loopVar3]*features[strongClassifier[0][loopVar3]]) <= (strongClassifier[1][loopVar3]*strongClassifier[2][loopVar3]):				# Find the decision
						weightedDecision+=strongClassifier[3][loopVar3]*1	# Find weighted decision
				
			if weightedDecision >= strongClassifier[5]:# If weighted decision is greater than strong classifier's threshold
				stageLevelDecisions.append(1)	# Declare the image to be Positive at this stage
			else:					# Else, 
				stageLevelDecisions.append(0)	# Declare the image to be negative at this stage
		
		eachStageDecisions.append(stageLevelDecisions)  # Save all the stagelevel decisions

		if all(stageLevelDecisions):			# If all stages said, "Positive", declare positive
			FalsePositives+=1	
			classifiedLabels.append(1)
		else:						# Otherwise, declare negative
			TrueNegatives+=1	
			classifiedLabels.append(0)
			correctClassifications+=1	
	
	
	#--------------------------------------------------------------------------------------------#
	# This section of the code computes all performance metrics (global and stage-wise) 
	eachStageDecisions=array(eachStageDecisions)
	validity=ones((eachStageDecisions.shape[0],1))
	eachStageDecisions=hstack((eachStageDecisions, validity))
	
	#print eachStageDecisions
	TP=zeros((1, eachStageDecisions.shape[1]-1))[0].tolist()
	FN=zeros((1, eachStageDecisions.shape[1]-1))[0].tolist()
	FP=zeros((1, eachStageDecisions.shape[1]-1))[0].tolist()
	TN=zeros((1, eachStageDecisions.shape[1]-1))[0].tolist()
	
	for loopVar2 in range(eachStageDecisions.shape[1]-1): # For every Stage
		subNumberofPos=((eachStageDecisions[0:NUMBER_OF_POS_IMAGES, eachStageDecisions.shape[1]-1]).tolist()).count(1)
		subNumberofNeg=((eachStageDecisions[NUMBER_OF_POS_IMAGES:eachStageDecisions.shape[0], eachStageDecisions.shape[1]-1]).tolist()).count(1)
		print 'Images Passed to Next Stage:', subNumberofPos, subNumberofNeg
		for loopVar1 in range(eachStageDecisions.shape[0]): # For every Image
			if eachStageDecisions[loopVar1, eachStageDecisions.shape[1]-1]==1:
				if eachStageDecisions[loopVar1, loopVar2]==1:
					if loopVar1<NUMBER_OF_POS_IMAGES:
						TP[loopVar2]+=1
					else:
						FP[loopVar2]+=1
				else:
					if loopVar1<NUMBER_OF_POS_IMAGES:
						FN[loopVar2]+=1
					else:
						TN[loopVar2]+=1
					eachStageDecisions[loopVar1, eachStageDecisions.shape[1]-1]=0
		if subNumberofPos!=0:	
			if loopVar2==0:
				TP[loopVar2]=TP[loopVar2]/float(subNumberofPos)
				FN[loopVar2]=FN[loopVar2]/float(NUMBER_OF_POS_IMAGES)
			else:
				TP[loopVar2]=TP[loopVar2-1]*(TP[loopVar2]/float(subNumberofPos))
				FN[loopVar2]=((FN[loopVar2-1]*NUMBER_OF_POS_IMAGES)+FN[loopVar2])/float(NUMBER_OF_POS_IMAGES)
		else:
			TP[loopVar2]=TP[loopVar2-1]
			FN[loopVar2]=FN[loopVar2-1]

		if subNumberofNeg!=0:
			if loopVar2==0:	
				FP[loopVar2]=FP[loopVar2]/float(subNumberofNeg)
				TN[loopVar2]=TN[loopVar2]/float(NUMBER_OF_NEG_IMAGES)
			else:
				FP[loopVar2]=FP[loopVar2-1]*(FP[loopVar2]/float(subNumberofNeg))
				TN[loopVar2]=((TN[loopVar2-1]*NUMBER_OF_NEG_IMAGES)+TN[loopVar2])/float(NUMBER_OF_NEG_IMAGES)
		else:
			FP[loopVar2]=FP[loopVar2-1]
			TN[loopVar2]=TN[loopVar2-1]		
	
	subNumberofPos=((eachStageDecisions[0:NUMBER_OF_POS_IMAGES, eachStageDecisions.shape[1]-1]).tolist()).count(1)
	subNumberofNeg=((eachStageDecisions[NUMBER_OF_POS_IMAGES:eachStageDecisions.shape[0], eachStageDecisions.shape[1]-1]).tolist()).count(1)
	print 'Images Passed to Next Stage:', subNumberofPos, subNumberofNeg
	#--------------------------------------------------------------------------------------------#
	
	# Compute overall accuracy and TP, FN, FP, TN rates
	accuracy=correctClassifications/float(NUMBER_OF_POS_IMAGES+NUMBER_OF_NEG_IMAGES)	
	FinalTP=TruePositives/float(NUMBER_OF_POS_IMAGES)
	FinalFN=FalseNegatives/float(NUMBER_OF_POS_IMAGES)
	FinalFP=FalsePositives/float(NUMBER_OF_NEG_IMAGES)
	FinalTN=TrueNegatives/float(NUMBER_OF_NEG_IMAGES)

	return classifiedLabels, accuracy, FinalTP, FinalFN, FinalFP, FinalTN, [TP, FN, FP, TN]  # Return the classification results

#----------------------------------------------------------------------------------------------#
# Main Code starts

global haarFeatures, sampleWeights		# Declare global HAAR feature matrix and weights array
haarFeatures=zeros((NUMBER_OF_TRAIN_POS_IMAGES+NUMBER_OF_TRAIN_NEG_IMAGES, 166001), dtype=float32)
sampleWeights=zeros((NUMBER_OF_TRAIN_POS_IMAGES+NUMBER_OF_TRAIN_NEG_IMAGES, 1))#, dtype=float64)
print shape(haarFeatures), shape(sampleWeights), shape(sampleWeights[:, :])
	
readImageHaar(folder+'train/', NUMBER_OF_TRAIN_POS_IMAGES, NUMBER_OF_TRAIN_NEG_IMAGES, 0) # Read the training images and compute HAAR features
finalCascadeClassifier=learnCascadeClassifier()#'''		# Learnt the cascade classifier
print '******************* Done Learning Cascade Classifier **************************'

# Classify the test data
classifiedLabels, accuracy, TP, FN, FP, TN, stageLevelScores = classifyTestImages(finalCascadeClassifier, folder+'test/', NUMBER_OF_TEST_POS_IMAGES, NUMBER_OF_TEST_NEG_IMAGES, TEST_POS_IMG_NUMBERING_OFFSET, TEST_NEG_IMG_NUMBERING_OFFSET)

# Print the test results
print '*********************** Test Results ********************************'
print 'Accuracy=', accuracy
print 'TP=', TP
print 'FN=', FN
print 'FP=', FP
print 'TN=', TN
print 'stageLevelScores=', array(stageLevelScores)
print '*********************** All Done ********************************'
