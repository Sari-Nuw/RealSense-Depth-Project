import csv
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from shapely.geometry import Polygon, LineString
from shapely import get_coordinates
from Sorting_utilities import coordinate_iou

#Getting pixel sizes across differnet points of the cluster
def cluster_sizing(bounding,x_diff,y_diff,poly,image,numbering):

    #To save the size of each segment
	segments = []

	#Defining the edges of the bounding box
	minx = bounding[0]
	miny = bounding[1]
	maxx = bounding[2]
	maxy = bounding[3]

	#Defining the horizontal and vertical lines splitting the cluster into thirds
	#Left Vertical line
	vert_left = [[minx+int(x_diff/4),miny],[minx+int(x_diff/4),maxy]]
	vert_line_left = LineString([vert_left[0],vert_left[1]])
	segment,numbering = line_clip(vert_line_left,poly,image, numbering)
	segments.append(segment)
	#Middle Vertical line
	vert_mid = [[minx+int(x_diff/2),miny],[minx+int(x_diff/2),maxy]]
	vert_line_mid = LineString([vert_mid[0],vert_mid[1]])
	segment,numbering = line_clip(vert_line_mid,poly,image,numbering)
	segments.append(segment)
	#Right Vertical Line
	vert_right = [[minx+int(3*x_diff/4),miny],[minx+int(3*x_diff/4),maxy]]
	vert_line_right = LineString([vert_right[0],vert_right[1]])
	segment,numbering = line_clip(vert_line_right,poly,image,numbering)
	segments.append(segment)
	#Top Horizontal line
	horz_top = [[minx,miny+int(y_diff/4)],[maxx,miny+int(y_diff/4)]]
	horz_line_top = LineString([horz_top[0],horz_top[1]])
	segment,numbering = line_clip(horz_line_top,poly,image,numbering)
	segments.append(segment)
	#Middle Horizontal Line
	horz_mid = [[minx,miny+int(y_diff/2)],[maxx,miny+int(y_diff/2)]]
	horz_line_mid = LineString([horz_mid[0],horz_mid[1]])
	segment,numbering = line_clip(horz_line_mid,poly,image,numbering)
	segments.append(segment)
	#Bottom Horizontal Line
	horz_bot = [[minx,miny+int(3*y_diff/4)],[maxx,miny+int(3*y_diff/4)]]
	horz_line_bot = LineString([horz_bot[0],horz_bot[1]])
	segment,numbering = line_clip(horz_line_bot,poly,image,numbering)
	segments.append(segment)

	return segments,numbering

#Establishing cluster_segments excel file
def establish_cluster_sizing(working_folder):
	with open(working_folder + '/Cluster_Sizing.csv', 'w',newline='') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file)
		#Writing the first row with all the headers
		writer.writerow(['Image #','Cluster #','Cluster Pixel Area','Absolute Cluster Area','Label','Cluster Height','Cluster Width','Absolute Height','Absolute Width','Vetical Left','Vertical Middle','Vertical Right','Horizontal Top','Horizontal Middle','Horizontal Bottom'])

#Writing information from precision_metrics to excel file
def establish_metrics(working_folder):
	with open(working_folder + '/Precision_Metrics.csv', 'w',newline='') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file)
		#Writing the first row with all the headers
		writer.writerow(['Image #','Metric Type','mAP','mAR','F1','True Positive [0.5:0.95]','False Positive [0.5:0.95]','False Negative [0.5:0.95]'])

# Making lengths of all polygon arrays equal
def equalize_polygons(polygons,polygons_info):

	#Finding longest set of polygons
	max_length = 0
	for poly in polygons:
		if len(poly) > max_length:
			max_length = len(poly)

	#Adding null points to make the list equal sizes
	i = 0
	for poly in polygons:
		while len(poly) < max_length:
			poly.append([0])
			polygons_info[i].append([0])
		i += 1

	return polygons, polygons_info

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
		
	return images_int

#Process annotation metrices for each image
def get_annotation_metrics(annotations,polygons):

	#Storing true positive, false positive, and false negative values at different iou threshold values
	TP_array = []
	FP_array = []
	FN_array = []

	#Iterating between iou_threshold of 0.5-0.95 to calculate the mAP,mAR, and F1-score
	iou_threshold = 0.5
	while iou_threshold < 1:
		#Track true positive detections
		TP = 0
		#To check whether an annotation/polygon have been recognized/matched 
		annotations_check = [False for _ in range(len(annotations))]
		i = 0
		for polygon in polygons:
			#To iterate across annotation check
			i = 0
			for annotation in annotations:
				iou = coordinate_iou(annotation,polygon)
				if iou >= iou_threshold:
					#Each annotation should only be detected once. Extra detections ae false positives
					if annotations_check[i] == False:
						TP += 1
						annotations_check[i] = True
				i += 1
	
		#False Negative. Annotation was never detected
		FN = len(annotations) - TP

		#False positive. Polygon was detected where there should not be a detection
		FP = len(polygons) - TP

		#Saving the metrics at this interval
		TP_array.append(TP)
		FP_array.append(FP)
		FN_array.append(FN) 

		iou_threshold += 0.05

	#Precison and recall arrays
	precision_array = []
	recall_array = []
	for i in range(len(TP_array)):
		precision_array.append(TP_array[i]/(TP_array[i]+FP_array[i]))
		recall_array.append(TP_array[i]/(TP_array[i]+FN_array[i]))

	#Mean average precision and mean average recall
	mAP = sum(precision_array)/len(precision_array)
	mAR = sum(recall_array)/len(recall_array)
	F1 = (2*mAP*mAR)/(mAP+mAR)

	return mAP, mAR, F1, TP_array, FP_array, FN_array

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
def line_setup(polygons,lines,img_size):

	#Sizing segmented areas from the images
	mask_sizes = []
	i = 0
	for polygon in polygons:
		#Getting area of each polygon
		if len(polygon) > 1:
			segment = round((Polygon(polygon).area/img_size*100),4)
			mask_sizes.append(segment)
		else:
			mask_sizes.append(0)
		i += 1

	#To store all the different lines based on total number of 'box points' found
	if lines == []:
		lines = ([[] for _ in range(len(polygons))])
	else:
		while len(lines) < len(mask_sizes):
			new_line = []
			while len(new_line) < len(lines[-1]):
				new_line.append(float('nan'))
			lines.append(new_line)

	#To update each line
	for i in range(len(mask_sizes)):
		if mask_sizes[i] == 0:
			lines[i].append(float('nan'))
		else:
			lines[i].append(mask_sizes[i])

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

	#Trimming empty sections of lines 
	for i in range(len(polygons)):
		number = False
		j = 0
		#Skipping over beggining sections of lines if they are emtpy (nan)
		while number == False and j < len(lines[i]):
			if ~np.isnan(lines[i][j]):
				number = True 
			j += 1

	return lines

#Conversion of pixel area to 'real' area using real substrate size
def pixel_absolute_area(pixel_area,averaged_length_pixels,substrate_real_size = 50):

	return pixel_area*substrate_real_size*substrate_real_size/(averaged_length_pixels*averaged_length_pixels)

#Plotting the growth curves
def plot_growth(polygons,lines):

	#Initializing x-axis
	x_axis = np.linspace(0,len(lines[-1]),num = len(lines[-1]))

	colors = cm.get_cmap('tab20', 20)

	fig, axs = plt.subplots()
	for i in range(len(polygons[-1])):
		axs.plot(x_axis,lines[i], label = 'Cluster {}'.format(i),color=colors(i))
	#Displaying the graphs
	axs.set_xlabel('Image')
	axs.set_ylabel('Relative Size by Pixel Number')
	axs.legend()
	plt.show()

#Isolate the cluster from the original image
def process_cluster(image_copy,poly,bounding,working_folder,i,j):

	#Get shape of the image
	height = image_copy.shape[0]
	width = image_copy.shape[1]

	minx = bounding[0]
	miny = bounding[1]
	maxx = bounding[2]
	maxy = bounding[3]
	
	#Limiting the upper boundaries to the maximum width and height
	limit_factor = 0.025
	limit_increase = 1 + limit_factor
	limit_decrease = 1 - limit_factor
	if maxy*limit_increase > height:
		uppery = height
	else:
		uppery = int(maxy*limit_increase)
	if maxx*limit_increase > width:
		upperx = width
	else:
		upperx = int(maxx*limit_increase)
	#Copying the cropped image section
	box_image = image_copy[int(miny*limit_decrease) : uppery, int(minx*limit_decrease) : upperx]
	#Converting from the polygon coordinates of the full picture to polygon coordinates in the box image
	local_poly = poly.copy()
	local_poly[:,0] = local_poly[:,0] - minx*limit_decrease
	local_poly[:,1] = local_poly[:,1] - miny*limit_decrease
	#Saving the bounded section of the image
	cv2.polylines(box_image, np.int32([local_poly]), True, (255, 0, 0), 5)

	#Saving isolated cluster image
	os.makedirs(working_folder + "/Cluster/",exist_ok=True)
	cv2.imwrite(working_folder + "/Cluster/image ({})_cluster ({}).JPG".format(i+1,j), cv2.cvtColor(box_image,cv2.COLOR_RGB2BGR))

	return box_image,local_poly

def save_annotation_image(img,working_folder,annotations,img_num):

	annotation_img = img.copy()
	for annotation in annotations[img_num]:
		cv2.polylines(annotation_img, np.int32([annotation]), True, (255, 0, 0), 5)

	#Saving image with annotations outlined
	cv2.imwrite(working_folder + "/Annotated/image ({}).JPG".format(img_num+1), cv2.cvtColor(annotation_img,cv2.COLOR_RGB2BGR))

#Saving the image information for an individual cluster in numpy array format
def save_cluster_array(sizing_image,poly,centre,box_image,local_poly,working_folder,i,j):
	cv2.polylines(sizing_image, np.int32([poly]), True, (255, 0, 0), 5)
	cv2.putText(sizing_image, str(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
	#Localizing polygon mask
	binary_mask = np.zeros((box_image.shape[0],box_image.shape[1]),int)
	for point in local_poly:
		binary_mask[point[1],point[0]] = 1
	array = [box_image[:,:,0],box_image[:,:,1],box_image[:,:,2],binary_mask]
	os.makedirs(working_folder + "/Arrays/",exist_ok=True)
	np.save(working_folder + "/Arrays/RGB_image ({})_cluster ({})".format(i+1,j),array)

#Saving the various image types
def save_image(working_folder,full_image,i):
	os.makedirs(working_folder + "/Picture/",exist_ok=True)
	cv2.imwrite(working_folder + "/Picture/images ({}).JPG".format(i+1), cv2.cvtColor(full_image,cv2.COLOR_RGB2BGR))

def save_sizing_image(working_folder,sizing_image,i):
	os.makedirs(working_folder + "/Sizing/",exist_ok=True)
	cv2.imwrite(working_folder + "/Sizing/images ({}).JPG".format(i+1), cv2.cvtColor(sizing_image,cv2.COLOR_RGB2BGR)) 

#Saving the image information in numpy array format
def save_image_array(full_image,polygon,working_folder,i):
	array = [full_image[:,:,0],full_image[:,:,1],full_image[:,:,2]]
	for poly in polygon:
		binary_mask = np.zeros((full_image.shape[0],full_image.shape[1]),int)
		for point in poly:
			if not isinstance(point,int):
				binary_mask[point[1],point[0]] = 1
		array.append(binary_mask)
	os.makedirs(working_folder + "/Arrays/",exist_ok=True)
	np.save(working_folder + "/Arrays/RGB_image ({})".format(i+1),array)

# Show results after filtering and before sorting
def save_unsorted_image(img,polygons,working_folder,img_num):

	#Copying image
	unsorted_img = img.copy()

	# Outling the polygons before sorting
	i = 0
	for poly in polygons[-1]:
		centre = Polygon(poly).centroid
		cv2.polylines(unsorted_img, np.int32([poly]), True, (255, 0, 0), 5)
		#cv2.putText(unsorted_img, 'Pred {} {}'.format(i,poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
		cv2.putText(unsorted_img, 'Pred {}'.format(i), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
		i += 1

	# Saving the image with unsorted polygons
	cv2.imwrite(working_folder + "/Unsorted/image ({}).JPG".format(img_num+1), cv2.cvtColor(unsorted_img,cv2.COLOR_RGB2BGR))

# Drawing harvested cluster outline
def visualising_harvested_clusters(harvested,full_image):

	if harvested != []:
		for poly in harvested:
			poly.reshape(-1,1,2)
			#Getting the centre point of the polygons
			centre = Polygon(poly).centroid
			#Saving the image with outlined clusters (harvested)
			cv2.polylines(full_image, np.int32([poly]), True, (0, 0, 255), 5)
			#cv2.putText(full_image, '{}'.format(poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)

#Writing information from cluster_segments to excel file
def write_cluster_sizing(segment,working_folder):
	with open(working_folder + '/Cluster_Sizing.csv', 'a',newline='') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file)
		#Writing new row
		if segment == []:
			writer.writerow('')
		else:
			writer.writerow(["img ({})".format(segment[0]),segment[1],segment[2],segment[3],segment[4][0],segment[4][1],segment[4][2],segment[4][3],segment[4][4],segment[5][0],segment[5][1],segment[5][2],segment[5][3],segment[5][4],segment[5][5]])

#Writing information from cluster_segments to excel file
def write_metrics(working_folder,metric_type,metrics,img_num):
	with open(working_folder + '/Precision_Metrics.csv', 'a',newline='') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file)
		#Writing new row
		writer.writerow(["img ({})".format(img_num+1),metric_type,metrics[0],metrics[1],metrics[2], metrics[3],metrics[4],metrics[5]])
		writer.writerow('')

