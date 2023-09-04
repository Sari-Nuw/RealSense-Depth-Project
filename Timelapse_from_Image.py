import cv2
import os
from MonocularDepthEstimation import *
from DepthMap import *
from PIL import Image

#Path to the images
image_folder = r'Timelapse\Exp3'

#To store the images
images = []

#Iterating through the folder to read the images (File needs to be formatted properly)
i = 1
for file in os.listdir(image_folder):
    img = cv2.imread(image_folder + '\img ({}).JPG'.format(i))
    images.append(img)
    i = i +1

#Saving the size of the image
height, width, layers = images[0].shape

i = 1
for img in images:
    #Getting estimated depth from image scaled between 0-1
    depth_frame = MonocularDepthEstimation(np.asarray(img),0,1)
    #Array of ones
    ones = np.ones((height,width))
    #Conversion of the estimated depth to RGB values (grayscale)
    depth = (ones-depth_frame)*255
    #Converting the depth frame type
    img = depth.astype(np.uint8)
    #Converting from array to image
    depth_img = Image.fromarray(img)
    #Saving the image
    depth_img.save(image_folder + '\Depth_Map {}.png'.format(i))
    print(i)
    i = i+1

