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
from shapely.geometry import Polygon, Point, mapping, LineString
from shapely.ops import split
from shapely import get_coordinates
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
test_images = working_folder + r"\Mushroom Pictures\Hungary Distance Pictures\Canon Pictures"
#test_images = working_folder + r"\Python_Midas\Timelapse\Experiment 3"
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
        if i < 32:
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

i = 0
######WRITING TO EXCEL SHEET
with open(results_folder + '\Cluster_Sizing.csv', 'w',newline='') as csv_file:
    #Creating the csv writer
    writer = csv.writer(csv_file)
    #Writing the first row with all the headers
    writer.writerow(['Image #','Cluster #','Vetical Left','Vertical Middle','Vertical Right','Horizontal Top','Horizontal Middle','Horizontal Bottom'])
    #Saving a copy of the current image to manipulate
#Displaying the images with outlines
    for polygon in polygons:
        img = np.copy(images[i])
        full_image = np.copy(images[i])
        #To track sizing lines
        numbering = 1
        #Polygons in each image
        j = 0
        for poly in polygon:
            #Draw lines
            if len(poly) > 1:
                poly.reshape(-1,1,2)
                #Estimating the avergae distance of the cluster from the camera
                polygon_coordinates = tuple(tuple(map(int,tup)) for tup in mapping(Polygon(poly))['coordinates'][0])
                #Getting the centre point of the polygons
                centre = Polygon(poly).centroid
                #Finding the bounding box of the polygon to save the image as its own unique section
                bounding = BoundingBox(poly)
                x_diff = bounding.maxx - bounding.minx
                y_diff = bounding.maxy - bounding.miny
                #Drawing the horizontal and vertical sizing lines on the image
                segments,numbering = cluster_sizing(bounding,x_diff,y_diff,poly,full_image,numbering)
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
                cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 10)
                cv2.putText(full_image, str(j), (int(centre.x),int(centre.y)-10), cv2.FONT_HERSHEY_COMPLEX, 5, (0,255,0), 8, cv2.LINE_AA)
                num = round((Polygon(poly).area/img_size*100),4)
                #cv2.putText(full_image, str(num), (int(centre.x),int(centre.y)+45), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 8, cv2.LINE_AA)
                cv2.imwrite(results_folder + "\Pictures\images ({})".format(i+1) + image_type, cv2.cvtColor(full_image,cv2.COLOR_RGB2BGR))
                #Writing to excel sheet
                to_write = ["img ({})".format(i+1),j,segments[0],segments[1],segments[2],segments[3],segments[4],segments[5]]
                writer.writerow(to_write)
            j += 1
        print(i)
        i += 1
        #Skip row between images
        writer.writerow('')