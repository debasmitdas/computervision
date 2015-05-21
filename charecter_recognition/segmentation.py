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

texture_method=0	# Flag to decide RGB or texture method
#----------------------------------------------------------------------------------------------#
#Filenames
if texture_method!=1:
	filename1='Image3.jpg'
	filename2='Image3_segmented.jpg'
	filename3='Image3_blue_segmented.jpg'
	filename4='Image3_green_segmented.jpg'
	filename5='Image3_red_segmented.jpg'
	filename7='Image3_segmented_rgb_contour.jpg'
	filename8='Image3_segmented_rgb_with_border.jpg'
	
if texture_method==1:
	filename1='pic1.jpg'
	filename2='pic1_segmented_texture.jpg'
	filename3='pic1_tex3_segmented.jpg'
	filename4='pic1_tex5_segmented.jpg'
	filename5='pic1_tex7_segmented.jpg'
	filename6='pic1_texture.jpg'
	filename7='pic1_segmented_texture_contour.jpg'
	filename8='pic1_segmented_texture_with_border.jpg'

#----------------------------------------------------------------------------------------------#
#This function implements otsu's algorithm. It takes in an image and min and max pixel values in it and returns the same image, optimal threshold and the histogram
def otsu(image, min_value, max_value):
	hist=cv.CreateHist([256], cv.CV_HIST_ARRAY, [[0,255]], 1) # Create a histogram variable
	cv.CalcHist([image], hist, 0, None) # Calculate the histogram
	threshold, variance=calc_threshold(hist, min_value, max_value) # Find the optimal threshold
	print threshold, variance
	return image, threshold,hist
#---------------------------------------------------------------------------------------------#	
# This function takes in a histogram and min and max value and returns the optimal threshold
def calc_threshold(hist, min_value, max_value):
	max_variance=0
	N=0.0
	for hist_bin in range(min_value, max_value+1): # Calculate N
		N+=cv.QueryHistValue_1D(hist,hist_bin)
	ipi=[]
	for hist_bin in range(min_value, max_value+1): # Calculate i*pi values
		ipi.append(hist_bin*cv.QueryHistValue_1D(hist,hist_bin)/N)
	mean_t=sum(ipi) # Calculate total mean
	
	w0=0
	sum_ipi=0
	for hist_bin in range(min_value, max_value+1): # For all possible thresholds
		w0=w0+cv.QueryHistValue_1D(hist,hist_bin)/N # Find w0
		w1=1-w0					    # Find w1
		sum_ipi=sum_ipi+(hist_bin*cv.QueryHistValue_1D(hist,hist_bin)/N) # Find sum of i*pi
		try:
			mean0=sum_ipi/w0			# Find mean of class 1
		except ZeroDivisionError:
			mean0=0					# Except when w0=0, then mean will be 0
		try:		
			mean1=(mean_t-sum_ipi)/w1		# Find mean of class 0
		except ZeroDivisionError:
			mean1=0					# Except when w1=0, then mean will be 0
		var_b=w0*w1*pow((mean0-mean1),2)		# Find between class variance
		if var_b>max_variance:				# Check if this is maximum variance so far
			max_variance=var_b			# If yes, save it
			best_threshold=hist_bin			# and remember the threshold
			
	return best_threshold, max_variance			# Return the best threshold and corresponding variance
#---------------------------------------------------------------------------------------------#	
# This function eliminates pixels from a grayscale image which are less than a given threshold
def conv_to_binary(image, threshold):
	for loopVar1 in range(0, image.height):		# Loop for all pixels
		for loopVar2 in range(0, image.width):
			if image[loopVar1,loopVar2]<threshold:	# If any pixel is less than threshold	
				image[loopVar1,loopVar2]=0	# Make it zero
			#else:					
			#	image[loopVar1,loopVar2]=255	
	return image
#---------------------------------------------------------------------------------------------#	
# This function merges three input images using logical AND or logical OR depending on whether its RGB or texture based method used
def merge(channels):
	merged_img=[] 
	merged_img = zeros((channels[0].height,channels[0].width,3), uint8) # Create a new image container
	for loopVar1 in range(0, merged_img.shape[0]):				# Loop for all pixels
		for loopVar2 in range(0, merged_img.shape[1]):
			if texture_method!=1:				
				if (channels[0][loopVar1, loopVar2]==0)or(channels[1][loopVar1, loopVar2]==0)or(channels[2][loopVar1, loopVar2]==0):						# If any of the channel is zero, the merged image will have zero pixel
					merged_img[loopVar1][loopVar2]=[255,255,255]
				
				else:			# If all of the channels are non-zero, then the merged image will have highest value
				#merged_img[loopVar1][loopVar2]=[channels[0][loopVar1, loopVar2], channels[1][loopVar1, loopVar2], channels[2][loopVar1, loopVar2]]
					merged_img[loopVar1][loopVar2]=[0, 0, 0]	
			else:                       
				if (channels[0][loopVar1, loopVar2]==0)and(channels[1][loopVar1, loopVar2]==0)and(channels[2][loopVar1, loopVar2]==0):					# If all of the channels are zero, the merged image will have highest pixel
					merged_img[loopVar1][loopVar2]=[255, 255, 255]

				else:		# If any of the channels are non-zero, then the merged image will have zero pixel
					merged_img[loopVar1][loopVar2]=[0,0,0]	
	return merged_img
#---------------------------------------------------------------------------------------------#
# This function finds the texture image of a given image with each of the variances as channels 
def find_texture_image(image):
	texture_img =[]
	texture_img=cv.CreateMat(image.height, image.width, cv.CV_8UC3) # Create a new image container
	for loopVar1 in range(0, texture_img.height):			# Loop for all pixels
		for loopVar2 in range(0, texture_img.width):
			var=[]
			windows=[3,5,7]
			for item in range(len(windows)):		# For each window-size
				center_of_op=(windows[item]/2)
				sample_pixels=[]
				for loopVar3 in range(-center_of_op, center_of_op+1): # Get all the pixels in the window
					for loopVar4 in range(-center_of_op, center_of_op+1):
						if ((loopVar1+loopVar3)>=0 and (loopVar1+loopVar3)<image.height) and ((loopVar2+loopVar4)>=0 and (loopVar2+loopVar4)<image.width):
							sample_pixels.append(image[(loopVar1+loopVar3),(loopVar2+loopVar4)])
				var.append(pow(std(sample_pixels),2))	# Find the variance of the pixels
			texture_img[loopVar1, loopVar2]=[var[0], var[1], var[2]] # Save all three variances
		print loopVar1
	return texture_img	# Return the texture image
#---------------------------------------------------------------------------------------------#
# This function takes in a binary segmented image and returns the contour image and also segmented version of the original image
def extractContour(segmented_image, orig_img):
	contour_img=[] 
	contour_img = zeros((segmented_image.shape[0],segmented_image.shape[1],1), uint8) # Create a new image container for contour image
	image_with_border=[]
	image_with_border=cv.CreateMat(orig_img.height, orig_img.width, cv.CV_8UC3) # Create a new image container for final image
	cv.Convert(orig_img, image_with_border) 					# Copy original image to this container
	for loopVar1 in range(0, segmented_image.shape[0]):		# Loop for all pixels
		for loopVar2 in range(0, segmented_image.shape[1]):
			if any(segmented_image[loopVar1][loopVar2])==True:	# For each non-zero pixel value
				try:
					# Check if any 4-connected pixel is zero, if yes, then make it 255 (border pixel)
					if (any(segmented_image[loopVar1-1][loopVar2])==False or any(segmented_image[loopVar1+1][loopVar2])==False or any(segmented_image[loopVar1][loopVar2-1])==False or any(segmented_image[loopVar1][loopVar2+1])==False):
						contour_img[loopVar1][loopVar2]=255
					else:	# If its not a border pixel, make it zero.
						contour_img[loopVar1][loopVar2]=0		
				except IndexError:	# If its a border pixel, then all 4 neighbors won't exist, then make it zero
					contour_img[loopVar1][loopVar2]=0
				image_with_border[loopVar1, loopVar2]=[255,255,255]	
			else:				# If its not a border pixel, make it zero in both contour and final image
				contour_img[loopVar1][loopVar2]=0
				image_with_border[loopVar1, loopVar2]=[0,0,0]
		print loopVar1					
	return contour_img, image_with_border	# Return contour and final images
#---------------------------------------------------------------------------------------------#	

#---------------------------------------------------------------------------------------------#	
def preprocess(orig_img):
	for loopVar1 in range(0, orig_img.height):
		for loopVar2 in range(0, orig_img.width):
			#print orig_img[loopVar1, loopVar2]
			if orig_img[loopVar1, loopVar2]==(255, 255, 255):
				orig_img[loopVar1, loopVar2]=[0,0,0]
		print loopVar1		
	return orig_img
#---------------------------------------------------------------------------------------------#	
# Main code
if texture_method==1:	# If texture based method, then find the texture image and save it as well as load the same image for next steps
	orig_img_gray = cv.LoadImage(filename1,0)
	texture_img=find_texture_image(orig_img_gray)
	cv.SaveImage(filename6, texture_img)
	orig_img = cv.LoadImage(filename6,1)
	orig_img_rgb=cv.LoadImage(filename1,1)
else:			# If RGB-based, then just the given image
	# Load input image
	orig_img = cv.LoadImage(filename1,1)
	#orig_img = preprocess(orig_img)
	#cv.ShowImage("preprocessed", orig_img)

channels=[]
for depth in range(orig_img.nChannels):		# Create image containers for all 3 channels
	channels.append(cv.CreateImage((orig_img.width, orig_img.height), orig_img.depth, 1))

cv.Split(orig_img, channels[0], channels[1], channels[2], None)	# Split the color image into three channels
#cv.ShowImage('Blue', channels[0]) # Show the matched image
#cv.ShowImage('Green', channels[1]) # Show the matched image
#cv.ShowImage('Red', channels[2]) # Show the matched image

min_value=0
max_value=255	
binary_imgs=[]
for loopVar1 in range(len(channels)):	# For each of the channel
	channels[loopVar1],th,hist=otsu(channels[loopVar1], min_value, max_value) # Find best threshold using otsu
	# Run multiple times if required
	if texture_method!=1:
		if loopVar1!=0:
			channels[loopVar1],th,hist=otsu(channels[loopVar1], 0, th)
			channels[loopVar1],th,hist=otsu(channels[loopVar1], 0, th)	
			pass
	else:
		channels[loopVar1],th,hist=otsu(channels[loopVar1], 0, th)
		#channels[loopVar1],th,hist=otsu(channels[loopVar1], 0, th)		
		pass
		
	# Plot the histogram for analysis	
	hist_data=[]
	for hist_bin in range(min_value, max_value+1):
		hist_data.append(cv.QueryHistValue_1D(hist,hist_bin))
	plt.plot(hist_data)
	plt.ylabel('frequencies')
	plt.show()
	
	# Convert them to binary images using the obtained threshold
	bin_img=cv.CreateMat(channels[loopVar1].height, channels[loopVar1].width, cv.CV_8UC1)
	cv.Convert(channels[loopVar1], bin_img)
	binary_imgs.append(conv_to_binary(bin_img, th))

cv.ShowImage('Blue_segmented', binary_imgs[0]) # Show the segmented image for blue channel or 3x3 variance channel
cv.ShowImage('Green_segmented', binary_imgs[1]) # Show the segmented image for green channel or 5x5 variance channel
cv.ShowImage('Red_segmented', binary_imgs[2]) # Show the segmented image for red channel or 7x7 variance channel


segmented_image=merge(binary_imgs) 		# Merge all three channels
cv2.imshow('Binary_segmented', segmented_image) # Show the Binary segmented image image
if texture_method==1:
	orig_img = orig_img_rgb
	
contour_image, image_with_border=extractContour(segmented_image, orig_img) # Extract contour and Final segmented images
cv2.imshow('Contour_Image', contour_image) # Show the Contour image
cv.ShowImage('Image_with_border', image_with_border) # Show the Final segmented image

cv2.imwrite(filename2, segmented_image) #Save the result	
cv.SaveImage(filename3, binary_imgs[0]) #Save the result	
cv.SaveImage(filename4, binary_imgs[1]) #Save the result	
cv.SaveImage(filename5, binary_imgs[2]) #Save the result
cv2.imwrite(filename7, contour_image) #Save the result
cv.SaveImage(filename8, image_with_border) #Save the result'''
cv2.waitKey(0) #Wait for key-press
