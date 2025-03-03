import cv2
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from shapely.geometry import Polygon, LineString, Point
from shapely import get_coordinates


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

	# #Trimming empty sections of lines 
	# for i in range(len(polygons)):
	# 	number = False
	# 	j = 0
	# 	#Skipping over beggining sections of lines if they are emtpy (nan)
	# 	while number == False and j < len(lines[i]):
	# 		if ~np.isnan(lines[i][j]):
	# 			number = True 
	# 		j += 1

	return lines

#Conversion of pixel area to 'real' area using real substrate size
def pixel_absolute_area(pixel_area,averaged_length_pixels,substrate_real_size = 50):

	return pixel_area*substrate_real_size*substrate_real_size/(averaged_length_pixels*averaged_length_pixels)

#Plotting the growth curves
def plot_growth(polygons,lines,working_folder):

	#Initializing x-axis
	x_axis = np.linspace(0,len(lines[-1]),num = len(lines[-1]))

	colors = cm.get_cmap('tab20', 20)

	fig, axs = plt.subplots()
	for i in range(len(polygons[-1])):
		axs.plot(x_axis,lines[i], label = 'Cluster {}'.format(i),color=colors(i))
	#Displaying the graphs
	axs.set_xlabel('Image Number')
	axs.set_ylabel('Relative Size by Pixel Number')
	axs.legend()
	plt.savefig(working_folder + 'Cluster Growth Curves.png')
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
	box_image_copy = box_image.copy()
	#Converting from the polygon coordinates of the full picture to polygon coordinates in the box image
	local_poly = poly.copy()
	local_poly[:,0] = local_poly[:,0] - minx*limit_decrease
	local_poly[:,1] = local_poly[:,1] - miny*limit_decrease
	# Creating polygon object to check mask
	local_polygon = Polygon(local_poly)
	#Localizing polygon mask
	binary_mask = np.zeros((box_image.shape[0],box_image.shape[1]),int)
	#check if each element of image is in the polygon
	for height in range(box_image.shape[0]):
		for width in range (box_image.shape[1]):
			if local_polygon.contains(Point(width,height)):
				binary_mask[height,width] = 1
			else:
				box_image_copy[height,width] = [0,0,0]

	#Saving the bounded section of the image
	cv2.polylines(box_image, np.int32([local_poly]), True, (255, 0, 0), 10)

	#Saving isolated cluster image
	os.makedirs(working_folder + "/Cluster/",exist_ok=True)
	cv2.imwrite(working_folder + "/Cluster/image ({})_cluster ({}).JPG".format(i+1,j), cv2.cvtColor(box_image,cv2.COLOR_RGB2BGR))
	cv2.imwrite(working_folder + "/Cluster/cropped_image ({})_cluster ({}).JPG".format(i+1,j), cv2.cvtColor(box_image_copy,cv2.COLOR_RGB2BGR))

	return box_image,local_poly

# def save_annotation_image(img,working_folder,annotations,img_num):

# 	annotation_img = img.copy()
# 	i = 0
# 	for annotation in annotations[img_num]:
# 		centre = Polygon(annotation).centroid
# 		cv2.polylines(annotation_img, np.int32([annotation]), True, (255, 0, 0), 10)
# 		cv2.putText(annotation_img, '{}'.format(i), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)
# 		i += 1

# 	#Saving image with annotations outlined
# 	cv2.imwrite(working_folder + "/Annotated/image ({}).JPG".format(img_num+1), cv2.cvtColor(annotation_img,cv2.COLOR_RGB2BGR))

def save_annotation_image(img,working_folder,annotations,img_num):

	annotation_img = img.copy()
	for annotation in annotations[img_num]:
		id = annotation[1]
		annotation = annotation[0]
		centre = Polygon(annotation).centroid
		cv2.polylines(annotation_img, np.int32([annotation]), True, (0, 255, 0), 2)
		#cv2.putText(annotation_img, '{}'.format(id), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)

	#Saving image with annotations outlined
	cv2.imwrite(working_folder + "/Annotated/image ({}).JPG".format(img_num+1), cv2.cvtColor(annotation_img,cv2.COLOR_RGB2BGR))

#Saving the image information for an individual cluster in numpy array format
def save_cluster_array(sizing_image,poly,centre,box_image,local_poly,working_folder,i,j):
	cv2.polylines(sizing_image, np.int32([poly]), True, (255, 0, 0), 10)
	cv2.putText(sizing_image, str(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)
	#Localizing polygon mask
	binary_mask = np.zeros((box_image.shape[0],box_image.shape[1]),int)
	for point in local_poly:
		binary_mask[point[1],point[0]] = 1
	# print('cluster')
	# plt.imshow(binary_mask)
	# plt.show()
	array = [box_image[:,:,0],box_image[:,:,1],box_image[:,:,2],binary_mask]
	#plt.imshow(array)
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
		cv2.polylines(unsorted_img, np.int32([poly]), True, (255, 0, 0), 10)
		cv2.putText(unsorted_img, 'Pred {} {}'.format(i,poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
		#cv2.putText(unsorted_img, 'Pred {}'.format(i), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)
		i += 1

	# Saving the image with unsorted polygons
	cv2.imwrite(working_folder + "/Unsorted/image ({}).JPG".format(img_num+1), cv2.cvtColor(unsorted_img,cv2.COLOR_RGB2BGR))





