import numpy as np
from shapely.geometry import Polygon

#Calculating intersection over union for coordiante list 
def coordinate_iou(poly,base):
    # print('iou')
    # print(poly)
    # print(base)
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
            # print('list')
            # print(polygon)
            # print(polygon[0])
            # print(base)
            # print(base[0])
            if not isinstance(base[0],int):
                if len(base[0]) > 1:
                    # print('polygon')
                    # print(polygon)
                    # print('base')
                    # print(base)
                    poly_iou = coordinate_iou(polygon[0],base[0])
                    if poly_iou > iou_max:
                        iou_max = poly_iou
                        #if check_size(polygon,base):
                        best_fit = polygon        

        #Setting best fit box 
        if (iou_max) >= 0.25:
            temp.append(best_fit)
        else:
            temp.append([[0],[0]])

    #Adding possible old boxes to the temporary baseline
    i = 0
    for poly in temp:
        #If bounding box is not empty baseline is not required
        if len(poly[0]) > 1:
            temp_baseline.append([[0],[0]])
        else:
            # print('baseline')
            # print(baseline[i])
            temp_baseline.append(baseline[i])
        i += 1
    
    #Checking to see if an old box has returned
    for polygon in polygons:
        iou_max = 0
        i = 0
        for previous in temp_baseline:
            if len(previous[0]) > 1:
                poly_iou = coordinate_iou(polygon[0],previous[0])
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
                if len(polygon[0]) == len(base[0]):
                    if np.allclose(base[0],polygon[0]): 
                        included = True
            #Add the new box if it is not included
            if not included:
                temp.append(polygon) 
    
    return temp

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
    for idx in range(len(result[0])):
        for idy in range(idx+1,len(result[0])):
            # Check that both indexes are polygons (not empty)
            if len(result[idx][0]) > 1 and len(result[idy][0]) > 1:
                poly1 = result[idx][0]
                poly2 = result[idy][0]
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