import numpy as np
from shapely.geometry import Polygon, LineString
from shapely import get_coordinates
import cv2
import math
import torch
import os
import mmcv
from mmdet.apis import inference_detector,init_detector
from matplotlib import pyplot as plt
from mmdet.structures.det_data_sample import DetDataSample#			!!!!!!!!!!!!!!
from mmengine.structures.instance_data import InstanceData#			!!!!!!!!!!!!!!
import torchvision.ops.boxes as bops#			!!!!!!!!!!!!!!

def annotation_iou(annotations,poly,full_image,centre):
	iou_max = 0
	for annotation in annotations:
		iou = coordinate_iou(poly,annotation)
		if  iou > iou_max:
			iou_max = iou
	cv2.putText(full_image, str(iou_max)[:4], (int(centre.x),int(centre.y)-10), cv2.FONT_HERSHEY_COMPLEX, 5, (0,255,0), 8, cv2.LINE_AA)

#Calculating iou for bounding boxes
def box_iou(box_1,box_2):

    intersection_width = min(box_1[0],box_2[0]) - max(box_1[2],box_2[2])
    intersection_height = min(box_1[1],box_2[1]) - max(box_1[3],box_2[3])

    if intersection_height <= 0 or intersection_width <= 0:
        return 0
    
    intersection_area = intersection_width*intersection_height
    
    box_1_area = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
    box_2_area = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])

    union_area = box_1_area + box_2_area - intersection_area

    return intersection_area/union_area

#Checking if CUDA is available on the device running the program
def check_cuda():
	if torch.cuda.is_available():
		use_device = "cuda"
	else:
		use_device = "cpu"
	print(use_device)
	return use_device

def cluster_sizing(bounding,x_diff,y_diff,poly,image,numbering):

    #To save the size of each segment
    segments = []

    #Defining the horizontal and vertical lines splitting the cluster into thirds
    vert_left = [[bounding.minx+int(x_diff/4),bounding.miny],[bounding.minx+int(x_diff/4),bounding.maxy]]
    vert_line_left = LineString([vert_left[0],vert_left[1]])
    segment,numbering = line_clip(vert_line_left,poly,image, numbering)
    segments.append(segment)
    vert_mid = [[bounding.minx+int(x_diff/2),bounding.miny],[bounding.minx+int(x_diff/2),bounding.maxy]]
    vert_line_mid = LineString([vert_mid[0],vert_mid[1]])
    segment,numbering = line_clip(vert_line_mid,poly,image,numbering)
    segments.append(segment)
    vert_right = [[bounding.minx+int(3*x_diff/4),bounding.miny],[bounding.minx+int(3*x_diff/4),bounding.maxy]]
    vert_line_right = LineString([vert_right[0],vert_right[1]])
    segment,numbering = line_clip(vert_line_right,poly,image,numbering)
    segments.append(segment)
    horz_top = [[bounding.minx,bounding.miny+int(y_diff/4)],[bounding.maxx,bounding.miny+int(y_diff/4)]]
    horz_line_top = LineString([horz_top[0],horz_top[1]])
    segment,numbering = line_clip(horz_line_top,poly,image,numbering)
    segments.append(segment)
    horz_mid = [[bounding.minx,bounding.miny+int(y_diff/2)],[bounding.maxx,bounding.miny+int(y_diff/2)]]
    horz_line_mid = LineString([horz_mid[0],horz_mid[1]])
    segment,numbering = line_clip(horz_line_mid,poly,image,numbering)
    segments.append(segment)
    horz_bot = [[bounding.minx,bounding.miny+int(3*y_diff/4)],[bounding.maxx,bounding.miny+int(3*y_diff/4)]]
    horz_line_bot = LineString([horz_bot[0],horz_bot[1]])
    segment,numbering = line_clip(horz_line_bot,poly,image,numbering)
    segments.append(segment)

    return segments,numbering

#Calculating intersection over union for coordiante list 
def coordinate_iou(poly,base):
    poly1 = Polygon(poly).buffer(0)
    poly2 = Polygon(base).buffer(0)
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersect / union

## dict to MMDetection InstanceData class
def dict_to_instance_data(instance_dict):#			!!!!!!!!!!!!!!
    instance_data = InstanceData()
    for key, value in instance_dict.items():
        setattr(instance_data, key, value)
    return instance_data

## dict to MMDetection DetDataSample class
def dict_to_det_data_sample(data_dict):#			!!!!!!!!!!!!!!
    det_data_sample = DetDataSample()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, we assume it is an InstanceData
            setattr(det_data_sample, key, dict_to_instance_data(value))
        else:
            setattr(det_data_sample, key, value)
    return det_data_sample

def delete_low_confidence_predictions(result,confidence_score_threshold=0.5):#			!!!!!!!!!!!!!!
    result = result.cpu().numpy().to_dict()
    low_conf_indices = []
    ## find which predictions are of low classification/confidence score and keep their index for deletion
    for idx in range(len(result["pred_instances"]["scores"])):
        if result["pred_instances"]["scores"][idx]<confidence_score_threshold:
            low_conf_indices.append(idx)    

    ## delete from all components of the result variable the instances with low classification/confidence score
    result["pred_instances"]["bboxes"] = np.delete(result["pred_instances"]["bboxes"],low_conf_indices, axis=0)
    result["pred_instances"]["scores"] = np.delete(result["pred_instances"]["scores"],low_conf_indices, axis=0)
    result["pred_instances"]["masks"] = np.delete(result["pred_instances"]["masks"],low_conf_indices, axis=0)
    result["pred_instances"]["labels"] = np.delete(result["pred_instances"]["labels"],low_conf_indices, axis=0)

    return dict_to_det_data_sample(result)

#Check for overlapping predictions and remove the result with lower confidence
def delete_overlapping_with_lower_confidence(result,iou_threshold = 0.8):#			!!!!!!!!!!!!!!
    result = result.cpu().numpy().to_dict()
    to_delete = []
    ## iterate through all existing pairs of predictions
    for idx in range(len(result["pred_instances"]["bboxes"])):
        for idy in range(idx+1,len(result["pred_instances"]["bboxes"])):
            ## create the pair of bounding boxes to be examined
            box1 = torch.tensor([result["pred_instances"]["bboxes"][idx]], dtype=torch.float)
            box2 = torch.tensor([result["pred_instances"]["bboxes"][idy]], dtype=torch.float)
            ## calculate iou of bounding boxes pair
            iou = bops.box_iou(box1, box2)
            ## if iou is above a defined threshold the two prediction are referring to the same instance,
            ## so we find the one with the lower classification/confidence score and keep its index to be deleted
            if iou>iou_threshold:
                if result["pred_instances"]["scores"][idx]>result["pred_instances"]["scores"][idy]:
                    to_delete.append(idy)
                else:
                    to_delete.append(idx)
    
    ## delete from all components of the result variable the overlapping instances with classification/confidence score
    result["pred_instances"]["bboxes"] = np.delete(result["pred_instances"]["bboxes"],to_delete, axis=0)
    result["pred_instances"]["scores"] = np.delete(result["pred_instances"]["scores"],to_delete, axis=0)
    result["pred_instances"]["masks"] = np.delete(result["pred_instances"]["masks"],to_delete, axis=0)
    result["pred_instances"]["labels"] = np.delete(result["pred_instances"]["labels"],to_delete, axis=0)

    return dict_to_det_data_sample(result)

# #Check for overlapping predictions and remove the result with lower confidence
# def delete_overlapping_with_lower_confidence(result,iou_threshold = 0.8):

#     to_delete = []

#     ## iterate through all existing pairs of predictions
#     for idx in range(len(result)):
#         if idx+1 < len(result):
#             for idy in range(idx+1,len(result)):
#                 ## create the pair of bounding boxes to be examined
#                 box1 = result[idx][1]
#                 box2 = result[idy][1]
#                 ## calculate iou of bounding boxes pair
#                 iou = box_iou(box1, box2)
#                 ## if iou is above a defined threshold the two prediction are referring to the same instance,
#                 ## so we find the one with the lower classification/confidence score and keep its index to be deleted
#                 #print('check threshold')
#                 if iou>iou_threshold:
#                     #print('removed')
#                     if result[idx][0]>result[idy][0]:
#                         to_delete.append(idy)
#                     else:
#                         to_delete.append(idx)
    
#     ## delete from all components of the result variable the overlapping instances with classification/confidence score
#     for id in to_delete:
#         result[id][3] = []

#     return result

#Getting annotations from text file (JSON Coco format)
def get_annotations(text_file):
	#Pathway to json file
	with open(text_file, 'r') as openfile:
		# Reading from json file
		object = openfile.read()

	#To store the image_id of the polygons
	image_id = []

	#Extracting image id's from the json file
	index = 0
	while index < len(object):
		#Find each instance of "image_id":
		index = object.find('\"image_id\":', index)
		if index == -1:
			break
		#Skip to the number
		index = index + 11
		#Isolate and save the number
		split_index = object.find(',',index)
		id = object[index:split_index]
		image_id.append(int(id))

	#To store the polygon values linked to each image
	images = [[] for _ in range(max(image_id))]

	#Getting the polygon values
	index = 0
	#To iterate through the image_ids 
	id_count = 0 
	id_index = image_id[id_count] 
	broke = False
	while index < len(object):
		polygons = []
		current_id = id_index
		while id_index == current_id:
			#Finding the polygon segment
			index = object.find('\"segmentation\":[[', index)
			if index == -1:
				break
			index = index + 17
			#Isolating the segment
			split_index = object.find(']',index)
			bounding = object[index:split_index]
			bounding = bounding.split()
			polygons.append(bounding)
			#Iterating the id_count to know which image the polygon belongs to
			id_count += 1
			if id_count < len(image_id):
				id_index = image_id[id_count]
			else:
				broke = True
				id_index += 1
				break
		#Saving the polygons to the correct image number
		images[id_index-2].append(polygons)
		if broke:
			break

	#Converting the polygons from string to int and pairing each two x,y values
	images_int = []
	for image in images:
		polygons_int = []
		for polygons in image[0]:
			polygons = polygons[0]
			index = 0
			pair = []
			polygon_int = []
			while index < len(polygons):
				split_index = polygons.find(',',index)
				if split_index == -1:
					num = polygons[index:]
					index = len(polygons)
				else:
					num = polygons[index:split_index]
					index = split_index + 1
				pair.append(int(float(num)))
				if len(pair) == 2:
					polygon_int.append(pair)
					pair = []
			polygons_int.append(np.asarray(polygon_int))
		images_int.append(polygons_int)
		
	return(images_int)

#Getting image files
def get_test_set(test_set_path):
	test_set = sorted(os.listdir(test_set_path))
	test_set_length = len(os.listdir(test_set_path))
	test_set = []
	for i in range(test_set_length):
		test_set.append('img ({}).JPG'.format(i+1))
	return test_set

#Clipping sizing lines down to the proper size based on cluster
def line_clip(line,poly,image,numbering):

	#Find where the line intersects with thw polygon
	line_intersect = Polygon(poly).buffer(0).intersection(line)
	clipped = []
	#Check if theres multiple intersection segments
	if line_intersect.geom_type == 'GeometryCollection':
		for k in range(len(line_intersect.geoms)):
			#Only considering lines (points are removed)
			if line_intersect.geoms[k].geom_type == 'LineString':
				clipped.append(line_intersect.geoms[k])
		clipped = get_coordinates(clipped).astype(int)
	#Only one intersection segment
	else:
		clipped = get_coordinates(line_intersect).astype(int)

	#Draw the line segments on the image
	k = 0
	segment_lengths = []
	while k < len(clipped) and len(clipped) > 1:
		cv2.line(image,clipped[k],clipped[k+1],(255, 0, 0),10)
		clipped_length = math.dist(clipped[k],clipped[k+1])
		if len(clipped) > 2 :
			#Removing small lines
			if clipped_length > 20:
				#cv2.putText(image, str(clipped_length), (clipped[k][0],clipped[k][1]), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 8, cv2.LINE_AA)
				cv2.putText(image, str(numbering), (clipped[k][0],clipped[k][1]), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 8, cv2.LINE_AA)
				segment_lengths.append(clipped_length)
				numbering += 1
		else:
			#cv2.putText(image, str(clipped_length), (clipped[k][0],clipped[k][1]), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 8, cv2.LINE_AA)
			cv2.putText(image, str(numbering), (clipped[k][0],clipped[k][1]), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 8, cv2.LINE_AA)
			segment_lengths.append(clipped_length)
			numbering += 1
		k += 2
	return segment_lengths,numbering

#Gathering the information from individual clusters across images to be able to track their growth
def line_setup(polygons,img_size):

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

	return lines,x_axis

#Erode the image
def mask_erosion(cluster_depth_mask,kernel_size=21,iterations=3):
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 

    img_eroded = cv2.erode(cluster_depth_mask, kernel, iterations=iterations) 

    img_eroded[img_eroded == 0] = np.nan
    
    return img_eroded

def pixel_absolute_area(pixel_area,averaged_length_pixels,substrate_size):

	return pixel_area*substrate_size*substrate_size/(averaged_length_pixels*averaged_length_pixels)


#Plotting the growth curves
def plot_growth(polygons,x_axis,lines):
    fig, axs = plt.subplots()
    for i in range(len(polygons[-1])):
        axs.plot(x_axis,lines[i], label = 'Cluster {}'.format(i))
    #Displaying the graphs
    axs.set_xlabel('Image')
    axs.set_ylabel('Relative Size by Pixel Number')
    axs.legend()
    plt.show()
	
#Detecting and sizing the substrate in the image
def substrate_processing(substrate_model,test_set,test_set_path,working_folder):

	detected_length_pixels = []
	averaged_length_pixels = []

	i = 0
	for test_img in test_set:

		# load the image and color correct
		img = mmcv.imread(test_set_path + test_img)
		img = mmcv.image.bgr2rgb(img)

		substrate_img = img.copy()

		#Substrate segmentation inference
		substrate_result = inference_detector(substrate_model, img).pred_instances

		#Draw bounding boxes on images
		for result in substrate_result:
			sub_result = result["bboxes"].cpu().numpy()[0]
			cv2.rectangle(substrate_img,(int(sub_result[0]),int(sub_result[1])),(int(sub_result[2]),int(sub_result[3])),(0,0,255),5)

		#Save images
		os.makedirs(working_folder + "/Substrate/",exist_ok=True)
		cv2.imwrite(working_folder + "/Substrate/images ({}).JPG".format(i+1), cv2.cvtColor(substrate_img,cv2.COLOR_RGB2BGR))
		i += 1
				
		# collect substrate length data
		detected_length_pixels.append(substrate_result[0]["bboxes"].cpu().numpy()[0][2] - substrate_result[0]["bboxes"].cpu().numpy()[0][0])

		# calculate the substrate length average
		averaged_length_pixels.append(sum(detected_length_pixels)/len(detected_length_pixels))

	return averaged_length_pixels

