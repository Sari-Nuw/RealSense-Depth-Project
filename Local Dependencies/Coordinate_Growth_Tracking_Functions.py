import numpy as np
from shapely.geometry import Polygon, Point, mapping, LineString
from shapely import get_coordinates
import cv2
import math

#Calculating intersection over union for coordiante list 
def coordinate_iou(poly,base):
    poly1 = Polygon(poly)
    poly2 = Polygon(base)
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersect / union
    
#Using intersection over union method to track the same mushrooms for coordinate lists
def coordinate_sort(polygons,basis,baseline):
    temp = []
    temp_baseline = []
    #Iterate through the 'base polygons'
    for base in basis:
        #Set maximum iou to 0 (no intersection)
        iou_max = 0
        #Iterate through the next set of bounding boxes
        for polygon in polygons:
            #Looking through normal or empty boxes
            if len(base) > 1:
                poly_iou = coordinate_iou(polygon,base)
                if poly_iou > iou_max:
                    iou_max = poly_iou
                    #if check_size(polygon,base):
                    best_fit = polygon          

        #Setting best fit box 
        if (iou_max) >= 0.25:
            temp.append(best_fit)
        else:
            temp.append([0])

    #Adding possible old boxes to the temporary baseline
    i = 0
    for poly in temp:
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
                    best_fit = polygon
                    #to locate position of the old box
                    location = i
            i += 1
        if iou_max >= 0.15:
            temp[location] = best_fit

    if basis != []:
        for polygon in polygons:
            #Checking for boxes not yet included
            included = False 
            for base in temp:  
                if len(polygon) == len(base):
                    if np.allclose(base,polygon): 
                        included = True
            #Add the new box if it is not included
            if not included:
                temp.append(polygon) 
    
    return temp

#Drop maks from result array that overlap too much
def drop_overalapping_mask_polygon(result, iou_threshold=0.3):
    '''
        Normalize data using MinMax Normalization
        
            Input: 
                - (list of lists and arrays): output data structure from "inference_detector" function, containing bboxes, confidences and boolean arrays of detected masks
                
            Return: 
                - (list of lists and arrays): same data structure with overlapping detections being deleted
    '''
    to_drop = []
    ## iterate through all existing pairs of predictions
    for idx in range(len(result)):
        for idy in range(idx+1,len(result)):
            # Check that both indexes are polygons (not empty)
            if len(result[idx]) > 1 and len(result[idy]) > 1:
                poly1 = result[idx] 
                poly2 = result[idy]
                # calculate iou of the polygon pair
                iou = coordinate_iou(poly1,poly2)
                if iou>iou_threshold:
                    #Append to drop list if above iou threshold and not previously added
                    if idy not in to_drop:
                        to_drop.append(idy)

    #Deleting the results in the repeated indexes
    if to_drop:
        for index in to_drop:
            result[index] = []
        #print("Deleted indices: ", to_drop)
    return result

#Clipping sizing lines down to the proper size based on cluster
def line_clip(line,poly,image,numbering):

    #Find where the line intersects with thw polygon
    line_intersect = Polygon(poly).intersection(line)
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