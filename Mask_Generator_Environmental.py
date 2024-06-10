import os
import sys
import shutil
import torch
import pickle
import mmcv
from mmengine import Config
from mmdet.apis import inference_detector, show_result_pyplot,init_detector
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry import Polygon, Point, mapping
from concave_hull import concave_hull_indexes
from Coordinate_Growth_Tracking_Functions import *
from UliEngineering.Math.Coordinates import BoundingBox
from Segment_Crop import *
import math
import csv
from PIL import Image
from Depth_Estimation import *
from timeseries_preparation import *
import time

if torch.cuda.is_available():
     use_device = "cuda"
else:
     use_device = "cpu"

print(use_device)
     
    
#-----------------------------------------------
# Set-up local paths and files
#architecture_selected_cluster = "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco"
architecture_selected_cluster = "mask_rcnn_r50_fpn_1x_coco"
working_folder = r"C:\Users\nuway\OneDrive\Desktop\Realsense Project"
configs_folder = working_folder + r"\Python_Midas\configs\cluster"
results_folder = working_folder + r"\Python_Midas\Results"
#test_images = working_folder + r"\Mushroom Pictures\Hungary Distance Pictures\Canon Pictures"
test_images = working_folder + r"\Python_Midas\Timelapse\Experiment 3"
#Update image type if changing which images are being used
image_type = ".JPG"
#Control whether or not to run analysis for single mushrooms after the clusters
single_mushroom = False

# get only the jpg files in the test_images folder
test_set = sorted(os.listdir(test_images))
test_set = [x for x in test_set if x.endswith(image_type)]

# find the trained weights and the configuration of the model
files = os.listdir(configs_folder)
architecture_config_file = [x for x in files if x.endswith(".py") and architecture_selected_cluster in x][0]
architecture_pretrained_file = configs_folder + [x for x in files if x.endswith(".pth")][0]
config_save_filename = "custom_config_" + architecture_selected_cluster

# LOAD CONFIG FILE FROM CUSTOM SAVED .PY FILE AND CHANGE THE PRETRAINED WEIGHT PATH
cfg = mmcv.Config.fromfile(configs_folder + "\\" + config_save_filename + ".py")
cfg["load_from"] = configs_folder + "\\" + architecture_selected_cluster + "_BEST_mAP.pth"

# build the model from the config file and the checkpoint file that exists inside the config as a path
model = init_detector(cfg, cfg["load_from"], device=use_device)

print("Cluster Model Complete")

#-----------------------------------------------
## Inference on images

images = []
image_files = []
polygons = []
data = []
#Image size in pixels
img_size = 0

i = 1
for file in os.listdir(test_images):
    if i > 0:
        if i < 33:
            # read image
            img = cv2.imread(test_images +'\img ({})'.format(i) + image_type)
            image_files.append(test_images +'\img ({})'.format(i) + image_type)
            if img_size == 0:
                #To compare with cluster sizes for relative sizing (assuming image has 3 color channels)
                img_size = img.size/3
            img_data = Image.open(test_images +'\img ({})'.format(i) + image_type)._getexif()
            if not img_data:
                data.append('No Time Data')
            else:
                data.append(img_data[36867])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            # create save name 
            save_name = file.replace(image_type, ("_prediction"+image_type))
            # run the model on the image
            result = inference_detector(model, img)
            consolidated_result = np.zeros((img.shape[0],img.shape[1]))
            #show_result_pyplot(model, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), result, out_file = results_folder +'\image ({})_predictions'.format(i) + image_type,score_thr=0.9)
            # Removing the results with a confidence level <= 0.9
            for j in range (len(result[0][0])):
                if result[0][0][j][4] <= 0.9:
                    result[1][0][j] = []
                else:
                    consolidated_result = np.logical_or(consolidated_result,result[1][0][j])
            results = []
            # Converting from a boolean mask to a coordinate mask
            for masks in result[1][0]:
                if masks != []:
                    #Converting the true/false mask matrix in each result to a coordinate list 
                    points = np.argwhere(masks).tolist()
                    #Flipping across the coordinates from (y,x) to (x,y)
                    points = np.flip(points,1)
                    points = points[0::10]
                    #Finding the concave hull (outline) of the mask 
                    #Note: you can change the concativity of the hull to have a 'tighter' fit
                    #A tighter fit can result in errors occuring due to the polygon 'overlapping' itself
                    #This can be changed through the library (ctrl+click concave_hull_indexes)
                    hull = concave_hull_indexes(points)
                    #Appending the points that make the outline
                    results.append(points[hull])
            polygons.append(results)
            #print(len(result[0][0]))
            # visualize the results, save visualization, score_thr is to determine the minimum confidence of a prediction in order to be visualized, by default use 0.5
            #show_result_pyplot(model, img, result, out_file = results_folder +'\\' + save_name,score_thr=0.5)
            # save the output of the model with pickle, files can be from 200Mb 
            # up to 1Gb or even more, depending on how many predictions are present
            #save_data_with_pickle(results_folder + "\\" + test_img.replace(image_type, "_data.pkl") ,'result')
            #if cuda is used
            torch.cuda.empty_cache()
            print(i)
    i += 1

#Establishing baseline for sorting
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

i = 1
for result in polygons:
    result = drop_overalapping_mask_polygon(result)
    i += 1

#Adding null points to make the list equal sizes
for i in range(len(polygons)):
    while len(polygons[i]) < len(polygons[-1]):
        polygons[i].append([0])

print('# of Images')
print(len(images))
print('# of polygon_images')
print(len(polygons))

#Extracting the environmental variables from the csv files
temp,humidity,co2 = environmental_variable_prep(test_images,image_files)
print('env variables')
print(temp)
print(humidity)
print(co2)


i = 0
#Tracking which clusters are available from each image
cluster_track = [[] for _ in range(len(images))]
#Tracking average depth of individual clusters
#mask_depths = [[] for _ in range(len(polygons))]
#Displaying the images with outlines
for polygon in polygons:
    img = np.copy(images[i])
    full_image = np.copy(images[i])
    #Polygons in each image
    j = 0
    for poly in polygon:
        #Draw lines
        if len(poly) > 1:
            poly.reshape(-1,1,2)
            #Estimating the avergae distance of the cluster from the camera
            polygon_coordinates = tuple(tuple(map(int,tup)) for tup in mapping(Polygon(poly))['coordinates'][0])
            #avg_depth = Depth_Estimation(polygon_coordinates,images[i])
            ##mask_depths[i].append(avg_depth)
            #Getting the centre point of the polygons
            centre = Polygon(poly).centroid
            #Finding the bounding box of the polygon to save the image as its own unique section
            bounding = BoundingBox(poly)
            image_copy = (img, cv2.COLOR_RGB2BGR)[0]
            height = image_copy.shape[0]
            width = image_copy.shape[1]
            #Limiting the upper boundaries to the maximum width and height
            if bounding.maxy*1.025 > height:
                uppery = height
            else:
                uppery = int(bounding.maxy*1.025)
            if bounding.maxx*1.025 > width:
                upperx = width
            else:
                upperx = int(bounding.maxx*1.025)
            box_image = image_copy[int(bounding.miny*0.975) : uppery, int(bounding.minx*0.975) : upperx]
            #Saving the bounded section of the image
            cv2.imwrite(results_folder + "\Clusters\image ({})_cluster ({})".format(i,j) + image_type, cv2.cvtColor(box_image,cv2.COLOR_RGB2BGR))
            #Saving the image with outlined clusters
            #Copying image to prevent markings from showing
            #if j == 3 or j == 4 or j == 8:
            cv2.putText(full_image, str(j), (int(centre.x),int(centre.y)-10), cv2.FONT_HERSHEY_COMPLEX, 5, (0,255,0), 8, cv2.LINE_AA)
            cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 10)
            #cv2.putText(full_image, str(num), (int(centre.x),int(centre.y)+45), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 8, cv2.LINE_AA)
            #cv2.putText(full_image, str(avg_depth), (int(centre.x),int(centre.y)+25), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 8, cv2.LINE_AA)
            cv2.imwrite(results_folder + "\Pictures\images ({})".format(i+1) + image_type, cv2.cvtColor(full_image,cv2.COLOR_RGB2BGR))
            cluster_track[i].append(j)
        j += 1
    print(i)
    i += 1

#Sizing segmented areas from the images
mask_sizes = [[] for _ in range(len(polygons))]
i = 0
for polygon in polygons:
    for poly in polygon:
        #Getting area of each polygon
        if len(poly) > 1:
            p = Polygon(poly)
            segment = round((p.area/img_size*100),4)
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

#Initializing x-axis (/2 to convert Time to hours if in series)
x_axis = np.linspace(0,len(mask_sizes),num = len(mask_sizes))#/2

#Polynomial fitting of lines
polyfit_line = []
env_start = len(x_axis)
for line in lines:
    filtered_line = [x for x in line if (math.isnan(x) == False)]
    start = 0
    check = True
    #Skipping all initial nan values   
    while start < len(line) and math.isnan(line[start]):      
            start += 1
    #Making environmental variables start with the recognized images
    if start < env_start:
        env_start = start 
    print('lines: ',line)
    print('filtered: ',filtered_line)
    print('start: ', start)
    print('end:', start+len(filtered_line))
    #Polyfit line (polynomial values, start point on x-axis, end point on x-axis)
    if len(filtered_line) >= 1:
        polyfit_line.append([np.polyfit(x_axis[start:start+len(filtered_line)],filtered_line,deg=2),start,start+len(filtered_line)])
    else:
        polyfit_line.append([[],0,0])


#Trimming empty sections of lines 
for i in range(len(polygons[-1])):
    number = False
    j = 0
    #Skipping over beggining sections of lines if they are emtpy (nan)
    while number == False and j < len(lines[i]):
        if ~np.isnan(lines[i][j]):
            number = True 
        j += 1
    print('lines ', lines[i])
    #plt.plot(x_axis[polyfit_line[i][1]:polyfit_line[i][2]],np.polyval(polyfit_line[i][0],x_axis[polyfit_line[i][1]:polyfit_line[i][2]]), label = 'Cluster {}'.format(i))

with open(test_images + ' growth_data.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    header = ['']
    i = 0
    for line in lines:
        header.append('Cluster #{}'.format(i))
        i += 1
    writer.writerow(header)
    i = 0
    for poly in polygons:
        to_write = ["img ({})".format(i+1)]
        for line in lines:
            if not math.isnan(line[i]):
                to_write.append(line[i])
            else:
                to_write.append('')
        i += 1
        writer.writerow(to_write)

#plt.plot(x_axis,lines[0], label = 'Cluster {}'.format(0))

#Plotting environemntal variables
for i in range(3):
    fig, axs = plt.subplots()
    #fig.subplots_adjust(bottom=0.2, right=0.85)
    yaxs = axs.twinx()
    if i == 0:  
        for i in range(len(polygons[-1])):
            axs.plot(x_axis,lines[i], label = 'Cluster {}'.format(i))
        yaxs.set_ylabel('Temperature (C)')
        yaxs.plot(x_axis[env_start:],temp[env_start:],label = "Temperature")
    if i == 1:
        for i in range(len(polygons[-1])):
            axs.plot(x_axis,lines[i], label = 'Cluster {}'.format(i))
        yaxs.set_ylabel('Relative Humidity (%)')
        yaxs.plot(x_axis[env_start:],humidity[env_start:], label = "Relative Humidity")
    if i == 2:
        for i in range(len(polygons[-1])):
            axs.plot(x_axis,lines[i], label = 'Cluster {}'.format(i))
        yaxs.set_ylabel('CO2 Concentration (ppm)')
        yaxs.plot(x_axis[env_start:],co2[env_start:], label = "CO2 Concentration")
    #Displaying the graphs
    axs.set_xlabel('Image')
    axs.set_ylabel('Relative Size by Pixel Number')
    axs.legend()
    yaxs.legend()
    plt.show()