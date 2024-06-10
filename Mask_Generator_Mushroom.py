import os
import sys
import shutil
import torch
import pickle
import mmcv
#from mmengine import Config
from mmdet.apis import inference_detector, show_result_pyplot,init_detector
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from concave_hull import concave_hull_indexes
from Coordinate_Growth_Tracking_Functions import *
from MonocularDepthEstimation import *
from DepthMap import *
import time
from UliEngineering.Math.Coordinates import BoundingBox
import math

start_time = time.time()

# if torch.cuda.is_available():
#     use_device = "cuda"
# else:
use_device = "cpu"
    
#-----------------------------------------------
# Set-up local paths and files
#architecture_selected = "mask_rcnn_r50_fpn_1x_coco"
architecture_selected = "detectors_htc_r50_1x_coco"
working_folder = r"C:\Users\nuway\OneDrive\Desktop\Realsense Project"
configs_folder = working_folder + r"\Python Code 3.10\configs\single"
results_folder = working_folder + r"\Python Code 3.10\Pictures"
#test_images = working_folder + r"\Mushroom Pictures\Hungary Mushrooms"
#test_images = working_folder + r"\Workshop Trial\Small Picture Subset"
#test_images = working_folder + r"\Python Code 3.10\Timelapse\Exp1"
test_images = working_folder + r"\Python Code 3.10\Pictures"
#test_images = working_folder + r'\Workshop Trial\Timelapse'
#Update image type if changing which images are being used
image_type = ".JPG"

# get only the jpg files in the test_images folder
test_set = sorted(os.listdir(test_images))
test_set = [x for x in test_set if x.endswith(image_type)]

# find the trained weights and the configuration of the model
files = os.listdir(configs_folder)
architecture_config_file = [x for x in files if x.endswith(".py") and architecture_selected in x][0]
architecture_pretrained_file = configs_folder + [x for x in files if x.endswith(".pth")][0]
config_save_filename = "custom_config_" + architecture_selected

# LOAD CONFIG FILE FROM CUSTOM SAVED .PY FILE AND CHANGE THE PRETRAINED WEIGHT PATH
cfg = mmcv.Config.fromfile(configs_folder + "\\" + config_save_filename + ".py")
cfg["load_from"] = configs_folder + "\\" + architecture_selected + "_BEST_mAP.pth"

# build the model from the config file and the checkpoint file that exists inside the config as a path
model = init_detector(cfg, cfg["load_from"], device=use_device)

print("Model Complete")

#-----------------------------------------------
# USED IN ORDER TO SAVE THE PREDICTIONS OF THE MODELS IN THE FORMAT OF THE ORIGINAL VARIABLE CREATED
def save_data_with_pickle(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load_data_with_pickle(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v
            
## USAGE
# save_data_with_pickle("./test_pickle.pkl",'result') # arg1=path_to_savefile, arg2,3,4,...=names_of_global_variables_to_save
# load_data_with_pickle("./test_pickle.pkl")

#-----------------------------------------------
## Inference on images

images = []
images_copy = []
depths = []
polygons = []

i = 1
for file in os.listdir(test_images):
    if i > 0:
        #     if i < 90:
        # read image
        img = cv2.imread(test_images +'\complete img ({}) section (1)'.format(i) + image_type)
        if img is not None:
            images_copy.append(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            # create save name 
            save_name = file.replace(image_type, ("_prediction"+image_type))
            # run the model on the image
            result = inference_detector(model, img)
            for j in range (len(result[0][0])):
                if result[0][0][j][4] <= 0.9:
                    result[1][0][j] = []
                # else:
                #     print(result[0][0][j][4])

            results = []
            j = 0
            for masks in result[1][0]:
                if masks != []:
                    #Converting the true/false mask matrix in each result to a coordinate list 
                    points = np.argwhere(masks).tolist()
                    #Flipping across the coordinates from (y,x) to (x,y)
                    points = np.flip(points,1)
                    #Finding the concave hull of the mask (outline)
                    #Note: you can change the concativity of the hull to have a 'tighter' fit
                    #A tighter fit can result in errors occuring due to the polygon 'overlapping' itself
                    #This can be changed through the library (ctrl+click concave_hull_indexes)
                    hull = concave_hull_indexes(points)
                    #Appending the points that make the outline
                    results.append(points[hull])
                j += 1
            polygons.append(results)
            print(len(result[0][0]))
            # visualize the results, save visualization, score_thr is to determine the minimum confidence of a prediction in order to be visualized, by default use 0.5
            #show_result_pyplot(model, img, result, out_file = results_folder +'\\' + save_name,score_thr=0.5)
            # save the output of the model with pickle, files can be from 200Mb 
            # up to 1Gb or even more, depending on how many predictions are present
            #save_data_with_pickle(results_folder + "\\" + test_img.replace(image_type, "_data.pkl") ,'result')
            #if cuda is used
            # depth_frame = MonocularDepthEstimation(np.asarray(img),0,4)
            # depths.append(depth_frame)
            torch.cuda.empty_cache()
            print(i)
    i += 1
    # break



#Establishing baseline for sorting
#Finding starting frame where mushrooms are recognized
baseline = []
i = 0
while baseline == []:
    if polygons[i] != []:
        baseline = polygons[i].copy()
        start = i
    i += 1

#Sorting the bounding boxes for consistency 
for i in range(start,len(polygons)-1):
    print(i)
    polygons[i+1] = coordinate_sort(polygons[i+1],polygons[i],baseline)
    #Updating baseline
    for j in range(len(polygons[i+1])):
        if len(polygons[i+1][j]) > 1:
            if j < (len(baseline)):
                baseline[j] = polygons[i+1][j]
            else:
                baseline.append(polygons[i+1][j])

#Adding null points to make the list equal sizes
for i in range(len(polygons)):
    while len(polygons[i]) < len(polygons[-1]):
        polygons[i].append([0])

print(len(polygons))
print(len(images))
i = 0
#Displaying the images with outlines
for polygon in polygons:
    #Polygons in each image
    j = 0
    for poly in polygon:
        #Draw lines
        if len(poly) > 1:
            poly.reshape(-1,1,2)
            #Finding the centroid of the polygon to label each section appropriately
            centre = Polygon(poly).centroid
            #Finding the bounding box of the polygon to save the image as its own unique section
            # bounding = BoundingBox(poly)
            # height = images_copy[i].shape[0]
            # width = images_copy[i].shape[1]
            # #Limiting the upper boundaries to the maximum width and height
            # if bounding.maxy*1.01 > height:
            #     uppery = height
            # else:
            #     uppery = int(bounding.maxy*1.01)
            # if bounding.maxx*1.01 > width:
            #     upperx = width
            # else:
            #     upperx = int(bounding.maxx*1.01)
            # box_image = images_copy[i][int(bounding.miny*0.99) : uppery, int(bounding.minx*0.99) : upperx]
            # #Saving the bounded section of the image
            # cv2.imwrite(results_folder + "\img ({}) part ({}).JPG".format(i,j), box_image)
            #Drawing the outline and labelling the outlined section with its number
            cv2.polylines(images[i], np.int32([poly]), True, (255, 0, 0), 10)
            cv2.putText(images[i], str(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 5, cv2.LINE_AA)
        j += 1
    i += 1

for i in range(start,len(polygons)):
    plt.imshow(images[i])
    plt.title(i)
    plt.show()

#Cropping segmented areas from the images
mask_sizes = [[] for _ in range(len(polygons))]
i = 0
for polygon in polygons:
    for poly in polygon:
        #Getting area of each polygon
        if len(poly) > 1:
            p = Polygon(poly)
            segment = p.area
            #Finding the depths of each individual pixel to find total depth
            # total_depth = np.sum(depth_frame[i][poly])
            # #Calculating average depth
            # average_depth = total_depth/segment
            print(i)
            #print(average_depth)
            mask_sizes[i].append(segment)
        else:
            mask_sizes[i].append(0)
    i += 1

#To store all the different lines based on total number of 'box points' found
lines = [[] for _ in range(len(polygons[-1]))]
for i in range(len(mask_sizes)):
    for j in range(len(polygons[i])):
        if mask_sizes[i][j] == 0:
            lines[j].append(float('nan'))
        else:
            lines[j].append(mask_sizes[i][j])

#Iterating across each line
for line in lines:
    #Iterating across each point in the lines
    i = 0
    while i < len(line):
        base = line[i]
        #Skipping beggining section where the region of interest does not exist
        while math.isnan(base):
            i += 1
            if i >= len(line):
                break
            base = line[i]
        #Skipping sections where region of interest is present many times in a row
        while base >= 0:
            i += 1
            if i >= len(line):
                break
            base = line[i]
        #Counting how many times the region of interest was missed
        count = 1
        while math.isnan(base):
            i += 1
            if i >= len(line):
                break
            count += 1
            base = line[i]
        #Interpolating and assigning ther interpolated values to the empty sections
        if i <= len(line):
            interpolation = (base - line[i-(count)])/count
            j = 1
            while count > 1:
                line[i-j] = base - interpolation*j
                count -= 1
                j += 1

y_axis = np.arange(1,len(mask_sizes)+1)

print( time.time() - start_time)

for i in range(len(polygons[-1])):
    plt.plot(y_axis,lines[i], label = 'line {}'.format(i+1))

plt.legend()
plt.show()
