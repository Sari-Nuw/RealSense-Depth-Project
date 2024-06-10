import os
import torch
import mmcv
from mmengine import Config
from mmdet.apis import inference_detector, show_result_pyplot,init_detector
import numpy as np
import cv2
from shapely.geometry import Polygon, Point, mapping
from concave_hull import concave_hull_indexes
from Coordinate_Growth_Tracking_Functions import *
from Segment_Crop import *
from Depth_Estimation import *
from DepthMap import *
from scipy import stats
import time
from UliEngineering.Math.Coordinates import BoundingBox
import pickle
from sort import *

if torch.cuda.is_available():
     use_device = "cuda"
else:
    use_device = "cpu"
      
#-----------------------------------------------
# Set-up local paths and files
#architecture_selected_cluster = "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco"
architecture_selected_cluster = "mask_rcnn_r50_fpn_1x_coco"
working_folder = r"C:\Users\nuway\OneDrive\Desktop\Realsense Project"
configs_folder = working_folder + r"\Python_Midas\configs\cluster"
results_folder = working_folder + r"\Python_Midas\Results"
#test_images = working_folder + r"\Mushroom Pictures\Hungary Distance Pictures\Canon Pictures"
test_images = working_folder + r"\Python_Midas\Timelapse\Experiment 1"
depth_test_images = working_folder + r"\Mushroom Pictures\Hungary Distance Pictures\Stereo Depth Map"
#Update image type if changing which images are being used
image_type = ".JPG"

#Using stereo depth data if available 
stereo_on = False

#Pickling data
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

# get only the jpg files in the test_images folder
# test_set = sorted(os.listdir(test_images))
# test_set = [x for x in test_set if x.endswith(image_type)]

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

#Arrays to store: color images, stereo depth images, estimated depth images, cluster outlines, average_depths
images = []
depth_images = []
estimated_depth_images = []
polygons = []
img_avg_depth = []

i = 1
for file in os.listdir(test_images):
    if i > 0:
        if i < 500:
            total_start = time.time()
            # Read image
            img = cv2.imread(test_images +'\img ({})'.format(i) + image_type)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            # First run of the model on the image
            result = inference_detector(model, img)
            consolidated_result = np.zeros((img.shape[0],img.shape[1]))
            for j in range (len(result[0][0])):
                if result[0][0][j][4] <= 0.9:
                    result[1][0][j] = []
                else:
                    consolidated_result = np.logical_or(consolidated_result,result[1][0][j])
            #consolidated_img = img*np.expand_dims(consolidated_result,2)
            #Perform monocular depth estimation
            start = time.time()
            depth_frame = MonocularDepthEstimation(img, 0, 2000)
            end = time.time()
            print('depth estimation')
            print(end-start)
            estimated_depth_images.append(depth_frame)
            consolidated_depth_frame = depth_frame*consolidated_result
            #Read stereo depth map
            if stereo_on:
                depth_img = cv2.imread(depth_test_images +'\img ({}).png'.format(i))
                depth_images.append(DepthMaptoFrame(depth_img,0,2000))
            #Running inference detection on depth filtered images
            #ref_dist = 1000
            #ref_dist = np.average(depth_frame).astype(int)
            consolidated_ref_dist = np.average(consolidated_depth_frame[consolidated_depth_frame > 0]).astype(int)
            std_dev = np.std(consolidated_depth_frame[consolidated_depth_frame > 0]).astype(int)
            consolidated_ref_dist = consolidated_ref_dist+2*std_dev
            print('deviation')
            print(std_dev)
            img_avg_depth.append(consolidated_ref_dist)
            threshold = depth_frame.copy()
            filtered_img = img#.copy()
            threshold[threshold <= consolidated_ref_dist] = 0
            filtered_img[threshold.astype(bool)] = 0
            result = inference_detector(model, filtered_img)  
            # Removing the results with a confidence level <= 0.9
            for j in range (len(result[0][0])):
                if result[0][0][j][4] <= 0.9:
                    result[1][0][j] = []
            results = []
            # Converting from a boolean mask to a coordinate mask
            start = time.time()
            for masks in result[1][0]:
                if masks != []:
                    #Converting the true/false mask matrix in each result to a coordinate list 
                    points = np.argwhere(masks).tolist()
                    #Flipping across the coordinates from (y,x) to (x,y)
                    points = np.flip(points,1)
                    #Finding the concave hull (outline) of the mask 
                    #Note: you can change the concativity of the hull to have a 'tighter' fit
                    #A tighter fit can result in errors occuring due to the polygon 'overlapping' itself
                    #This can be changed through the library (ctrl+click concave_hull_indexes)
                    #Using 1/20 points from the mask
                    points = points[0::20]
                    hull = concave_hull_indexes(points)
                    #Taking 1/10 points for the polygon outline
                    hull_array = points[hull][0::10]
                    #Appending the points that make the outline
                    results.append(hull_array)
            end = time.time()
            print('concave hull')
            print(end-start)   
            polygons.append(results)
            torch.cuda.empty_cache()
            print(i)
            print('total time')
            print(time.time()-total_start)
    i += 1

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

#Removing overlapping masks from inference results
i = 1
for result in polygons:
    result = drop_overalapping_mask_polygon(result)
    i += 1

i = 0
#Saving the images with outlines and estimated/stereo depth information
for polygon in polygons:
    #Polygons in each image
    #Copying the current image for processing
    img = np.copy(images[i])
    depth_img_copy = np.copy(estimated_depth_images[i])
    image_copy = (img, cv2.COLOR_RGB2BGR)[0]
    full_image = np.copy(images[i])
    to_pickle = []
    j = 0
    for poly in polygon:
        #Draw lines
        if len(poly) > 1:
            poly.reshape(-1,1,2)
            #Estimating the avergae distance of the cluster from the camera
            polygon_coordinates = tuple(tuple(map(int,tup)) for tup in mapping(Polygon(poly))['coordinates'][0])
            avg_depth = Depth_Estimation(polygon_coordinates,estimated_depth_images[i])
            #Getting the centre point of the polygons
            centre = Polygon(poly).centroid
            #Finding the bounding box of the polygon to save the image as its own unique section
            bounding = BoundingBox(poly)
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
            #Getting the box of the matching depth image
            depth_box_image = depth_img_copy[int(bounding.miny*0.975) : uppery, int(bounding.minx*0.975) : upperx]
            #Converting from the polygon coordinates of the full picture to polygon coordinates in the box image
            local_poly = poly.copy()
            local_poly[:,0] = local_poly[:,0] - bounding.minx*0.975
            local_poly[:,1] = local_poly[:,1] - bounding.miny*0.975
            #Saving the bounded section of the image
            cv2.polylines(box_image, np.int32([local_poly]), True, (255, 0, 0), 5)
            cv2.imwrite(results_folder + "\Depth Pictures\Clusters\image ({})_cluster ({})".format(i+1,j) + image_type, cv2.cvtColor(box_image,cv2.COLOR_RGB2BGR))
            #Saving the image with outlined clusters
            cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 5)
            cv2.putText(full_image, str(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
            if stereo_on:
                avg_stereo_depth = Stereo_Depth_Estimation(polygon_coordinates,depth_images[i])
                cv2.putText(full_image, str(avg_stereo_depth) + ' cm', (int(centre.x),int(centre.y)+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(full_image, str(avg_depth/img_avg_depth[i]*100)[:4], (int(centre.x),int(centre.y)+55), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
        j += 1
    cv2.imwrite(results_folder + "\Depth Pictures\images ({})".format(i+1) + image_type, cv2.cvtColor(full_image,cv2.COLOR_RGB2BGR))
    #depth_image = DepthMap(estimated_depth_images[i],0,2000)
    #cv2.imwrite(r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Mushroom Pictures\Hungary Distance Pictures\Estimated Depth Map\images ({})".format(i+1) + image_type, np.array(depth_image))
    #to_pickle = np.dstack((box_image,depth_box_image,binary_mask))
    binary_mask = np.zeros((box_image.shape[0],box_image.shape[1]),int)
    for point in local_poly:
        binary_mask[point[1],point[0]] = 1
    #to_pickle = np.dstack((to_pickle,binary_mask))
    to_pickle = np.dstack((box_image,depth_box_image,binary_mask))
    #print(to_pickle)
    save_data_with_pickle(results_folder + "\pickle\\image_pickle_{}.pkl".format(i+1),'to_pickle') # arg1=path_to_savefile, arg2,3,4,...=names_of_global_variables_to_save
    print(i)
    i += 1

#pickle uses the same name as the global variable for the pickle info -> info[i] = to_pickle to store all the information
# to_pickle = 0
# for i in range (4):
#     load_data_with_pickle(results_folder + "\pickle\\image_pickle_{}.pkl".format(i+1))
#     print(i)
#     print(to_pickle)