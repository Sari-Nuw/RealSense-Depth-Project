import numpy as np
from shapely.geometry import Polygon, LineString
from shapely import get_coordinates
import cv2
import math

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

#Calculating intersection over union for coordiante list 
def coordinate_iou(poly,base):
    poly1 = Polygon(poly).buffer(0)
    poly2 = Polygon(base).buffer(0)
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersect / union

#Using intersection over union method to track the same mushrooms for coordinate lists
def coordinate_sort(polygons,basis,polygons_info,baseline):
    temp = []
    temp_baseline = []
    #Iterate through the 'base polygons'
    for base in basis:
        #Set maximum iou to 0 (no intersection)
        iou_max = 0
        #Iterate through the next set of bounding boxes
        i = 0
        for polygon in polygons:
            #Looking through normal or empty boxes
            if len(base) > 1:
                poly_iou = coordinate_iou(polygon,base)
                if poly_iou > iou_max:
                    iou_max = poly_iou
                    #if check_size(polygon,base):
                    best_fit = [polygon,polygons_info[i]]  
            i += 1       

        #Setting best fit box 
        if (iou_max) >= 0.25:
            temp.append(best_fit)
        else:
            temp.append([[0],[0]])

    #Adding possible old boxes to the temporary baseline
    i = 0
    for poly in temp[0]:
        #If bounding box is not empty baseline is not required
        if len(poly) > 1:
            temp_baseline.append([0])
        else:
            temp_baseline.append(baseline[i])
        i += 1
    
    #Checking to see if an old box has returned
    for polygon in polygons:
        iou_max = 0
        i = 0
        for previous in temp_baseline:
            if len(previous) > 1:
                poly_iou = coordinate_iou(polygon,previous)
                if poly_iou > iou_max:
                    iou_max = poly_iou
                    best_fit = [polygon,polygons_info[i]]
                    #to locate position of the old box
                    location = i
            i += 1
        if iou_max >= 0.5:
            temp[location] = best_fit

    if basis != []:
        i = 0
        for polygon in polygons:
            #Checking for boxes not yet included
            included = False 
            for base in temp:
                if not isinstance(base[0],int): 
                    if len(polygon) == len(base[0]):   
                        if np.allclose(base[0],polygon): 
                            included = True
            #Add the new box if it is not included
            if not included:
                temp.append([polygon,polygons_info[i]]) 
            i += 1
    
    polygons_temp = [x[0] for x in temp]
    info_temp = [x[1] for x in temp]

    return polygons_temp,info_temp
    
#Using intersection over union method to track the same mushrooms for coordinate lists
def coordinate_sort_depth(polygons,basis,polygons_info,polygons_depth_info,baseline):
    temp = []
    temp_baseline = []
    #Iterate through the 'base polygons'
    for base in basis:
        #Set maximum iou to 0 (no intersection)
        iou_max = 0
        #Iterate through the next set of bounding boxes
        i = 0
        for polygon in polygons:
            #Looking through normal or empty boxes
            if len(base) > 1:
                poly_iou = coordinate_iou(polygon,base)
                if poly_iou > iou_max:
                    iou_max = poly_iou
                    #if check_size(polygon,base):
                    best_fit = [polygon,polygons_info[i],polygons_depth_info[i]]  
            i += 1       

        #Setting best fit box 
        if (iou_max) >= 0.25:
            temp.append(best_fit)
        else:
            temp.append([[0],[0],[0]])

    #Adding possible old boxes to the temporary baseline
    i = 0
    for poly in temp[0]:
        #If bounding box is not empty baseline is not required
        if len(poly) > 1:
            temp_baseline.append([0])
        else:
            temp_baseline.append(baseline[i])
        i += 1
    
    #Checking to see if an old box has returned
    for polygon in polygons:
        iou_max = 0
        i = 0
        for previous in temp_baseline:
            if len(previous) > 1:
                poly_iou = coordinate_iou(polygon,previous)
                if poly_iou > iou_max:
                    iou_max = poly_iou
                    best_fit = [polygon,polygons_info[i],polygons_depth_info[i]]
                    #to locate position of the old box
                    location = i
            i += 1
        if iou_max >= 0.5:
            temp[location] = best_fit

    if basis != []:
        i = 0
        for polygon in polygons:
            #Checking for boxes not yet included
            included = False 
            for base in temp:
                if not isinstance(base[0],int): 
                    if len(polygon) == len(base[0]):   
                        if np.allclose(base[0],polygon): 
                            included = True
            #Add the new box if it is not included
            if not included:
                temp.append([polygon,polygons_info[i],polygons_depth_info[i]]) 
            i += 1
    
    polygons_temp = [x[0] for x in temp]
    info_temp = [x[1] for x in temp]
    depth_info_temp = [x[2] for x in temp]

    return polygons_temp,info_temp,depth_info_temp

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
    while k < len(clipped):
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

#Check for overlapping predictions and remove the result with lower confidence
def delete_overlapping_with_lower_confidence(result,iou_threshold = 0.8):

    to_delete = []

    ## iterate through all existing pairs of predictions
    for idx in range(len(result)):
        if idx+1 < len(result):
            for idy in range(idx+1,len(result)):
                ## create the pair of bounding boxes to be examined
                box1 = result[idx][1]
                box2 = result[idy][1]
                ## calculate iou of bounding boxes pair
                iou = box_iou(box1, box2)
                ## if iou is above a defined threshold the two prediction are referring to the same instance,
                ## so we find the one with the lower classification/confidence score and keep its index to be deleted
                print('check threshold')
                if iou>iou_threshold:
                    print('removed')
                    if result[idx][0]>result[idy][0]:
                        to_delete.append(idy)
                    else:
                        to_delete.append(idx)
    
    ## delete from all components of the result variable the overlapping instances with classification/confidence score
    for id in to_delete:
        result[id][3] = []

    return result