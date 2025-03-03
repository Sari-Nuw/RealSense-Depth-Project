from shapely.geometry import Polygon
import numpy as np

#Sorting clusters for tracking
def cluster_sort(polygons,polygons_info,baseline):

	#Establishing baseline
	if baseline == []:
		#Check for polygons to set baseline
		if polygons[-1] != []:
			i = 0
			to_delete = []
			for polygon in polygons[-1]:
				#Check that recognised mushrooms are immature
				if polygons_info[0][i][0] == 0:
					baseline.append([polygon,polygons_info[0][i]])
				#Remove mature clusters
				else:
					to_delete.append(i)
				i += 1
			#New polygons/polygons info with mature clusters removed
			polygons_cropped = [[]]
			polygons_info_cropped = [[]]
			i = 0
			for base in baseline:
				if i not in to_delete:
					polygons_cropped[0].append(base[0])
					polygons_info_cropped[0].append(base[1])
				i += 1
		#Exit the function after establishing baseline or if no polygons
		return polygons_cropped,polygons_info_cropped,baseline,to_delete

	polygons[-1],polygons_info[-1], to_delete = polygon_sort(polygons[-1],polygons_info[-1],baseline)

	#Updating baseline
	for j in range(len(polygons[-1])):
		if len(polygons[-1][j]) > 1:
			if j < (len(baseline)):
				baseline[j] = (polygons[-1][j],polygons_info[-1][j])
			else:
				baseline.append([polygons[-1][j],polygons_info[-1][j]])

	return polygons,polygons_info,baseline,to_delete

#Sorting clusters for tracking
def cluster_sort_lab(polygons,polygons_info,baseline):

	#Establishing baseline
	if baseline == []:
		#Check for polygons to set baseline
		if polygons[-1] != []:
			i = 0
			for polygon in polygons[-1]:
				#baseline.append([polygon,polygons_info[0][i]])
				baseline.append([polygon,polygons_info[-1][i]])
				i += 1
		#Exit the function after establishing baseline or if no polygons
		return polygons,polygons_info,baseline

	polygons[-1],polygons_info[-1] = polygon_sort_lab(polygons[-1],polygons_info[-1],baseline)

	#Updating baseline
	for j in range(len(polygons[-1])):
		if len(polygons[-1][j]) > 1:
			if j < (len(baseline)):
				baseline[j] = (polygons[-1][j],polygons_info[-1][j])
			else:
				baseline.append([polygons[-1][j],polygons_info[-1][j]])

	return polygons,polygons_info,baseline

#Calculating intersection over union for coordiante list 
def coordinate_iou(poly,base):

	poly1 = Polygon(poly).buffer(0)
	poly2 = Polygon(base).buffer(0)

	#Getting intersection of both polygons
	intersect = poly1.intersection(poly2).area
	if intersect == 0:
		return 0
	
	#Getting union and intersection over union
	union = poly1.union(poly2).area
	iou = intersect / union

	return iou

#Using intersection over union method to track the same mushrooms for coordinate lists 
def polygon_sort(polygons,polygons_info,baseline,iou_baseline = 0.2):

	temp = [[] for _ in range(len(baseline))]
	included = []

	#Iterate through the 'baseline polygons'
	i = 0
	for base in baseline:
		#Set maximum iou to iou_baseline (minimum acceptable iou)
		iou_max = iou_baseline
		best_fit = 0
		location = -1
		#Iterate through the next set of bounding boxes
		j = 0
		for polygon in polygons:
			#Looking through normal or empty boxes
			if len(base[0]) > 1:
				poly_iou = coordinate_iou(polygon,base[0])
				if poly_iou > iou_max:
					iou_max = poly_iou
					best_fit = [polygon,polygons_info[j]]
					location = j
			j += 1   

		if location != -1:
			included.append(location)
		
		#Setting best fit box 
		if iou_max == 1:
			temp[i] = [[0],[0]]
		elif iou_max > iou_baseline:
			temp[i] = best_fit
		else:
			temp[i] = [[0],[0]]

		i += 1
		
	for i in range(len(polygons)):
		#Adding new polygons
		if i not in included:
			if polygons_info[i][0] == 0:
				temp.append([polygons[i],polygons_info[i]])

	polygons_temp = [x[0] for x in temp]
	info_temp = [x[1] for x in temp]

	to_delete = []

	i = 0
	for poly in polygons:
		included = False
		for temp in polygons_temp:
			if np.all(poly[0] == temp[0]):
				included = True
		if not included:
			to_delete.append(i)
		i += 1
	
	print('poly')
	for poly in polygons:
		print(poly[0])
	print('temp')
	for poly in polygons_temp:
		print(poly[0])
	print(to_delete)

	return polygons_temp, info_temp, to_delete

#Using intersection over union method to track the same mushrooms for coordinate lists 
def polygon_sort_lab(polygons,polygons_info,baseline,iou_baseline = 0.2):

	temp = [[] for _ in range(len(baseline))]
	included = []

	#Iterate through the 'baseline polygons'
	i = 0
	for base in baseline:
		#Set maximum iou to iou_baseline (minimum acceptable iou)
		iou_max = iou_baseline
		best_fit = 0
		location = -1
		#Iterate through the next set of bounding boxes
		j = 0
		for polygon in polygons:
			#Looking through normal or empty boxes
			if len(base[0]) > 1:
				poly_iou = coordinate_iou(polygon,base[0])
				if poly_iou > iou_max:
					iou_max = poly_iou
					best_fit = [polygon,polygons_info[j]]
					location = j
			j += 1   

		if location != -1:
			included.append(location)
		
		#Setting best fit box 
		if iou_max == 1:
			temp[i] = [[0],[0]]
		elif iou_max > iou_baseline:
			temp[i] = best_fit
		else:
			temp[i] = [[0],[0]]

		i += 1
		
	for i in range(len(polygons)):
		#Adding new polygons
		if i not in included:
			temp.append([polygons[i],polygons_info[i]])

	polygons_temp = [x[0] for x in temp]
	info_temp = [x[1] for x in temp]

	return polygons_temp, info_temp