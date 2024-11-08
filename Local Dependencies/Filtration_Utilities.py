import math
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.structures.instance_data import InstanceData
import numpy as np
from PIL import Image, ImageStat
import torch 
import torchvision.ops.boxes as bops
from shapely.geometry import Polygon

#Calculating iou for bounding boxes
def box_iou(box_1,box_2):

	#Getting intersection width and height
	intersection_width = min(box_1[2],box_2[2]) - max(box_1[0],box_2[0]) 
	intersection_height = min(box_1[3],box_2[3]) - max(box_1[1],box_2[1]) 

	#No intersection
	if intersection_height <= 0 or intersection_width <= 0:
		return 0

	#Caclulating intersection area and area of each box
	intersection_area = intersection_width*intersection_height

	box_1_area = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
	box_2_area = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])

	#One box completely within another box
	if box_1_area == intersection_area or box_2_area == intersection_area:
		return 1

	union_area = box_1_area + box_2_area - intersection_area

	return intersection_area/union_area

#Adjust image brightness for processing
def brightness(im_file):       
    im = Image.open(im_file)        
    stat = ImageStat.Stat(im)        
    r,g,b = stat.mean        
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

## dict to MMDetection InstanceData class
def dict_to_instance_data(instance_dict):
    instance_data = InstanceData()
    for key, value in instance_dict.items():
        setattr(instance_data, key, value)
    return instance_data

## dict to MMDetection DetDataSample class
def dict_to_det_data_sample(data_dict):
    det_data_sample = DetDataSample()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, we assume it is an InstanceData
            setattr(det_data_sample, key, dict_to_instance_data(value))
        else:
            setattr(det_data_sample, key, value)
    return det_data_sample

def delete_low_confidence_predictions(result,confidence_score_threshold=0.5):
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
def delete_overlapping_with_lower_confidence(result,iou_threshold = 0.2):
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

def expand_box(box, scale_factor=0.10):
    """
    Expand the bounding box by a given scale factor (default is 10%).

    Args:
    - box: (Tensor) Bounding box in format [x_min, y_min, x_max, y_max]
    - scale_factor: (float) Fraction by which to expand the bounding box.

    Returns:
    - expanded_box: (Tensor) The expanded bounding box.
    """
    # Calculate the width and height of the box
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Calculate the expansion in both width and height (10% by default)
    delta_w = width * scale_factor
    delta_h = height * scale_factor

    # Expand the box
    x_min_expanded = box[0] - delta_w / 2
    y_min_expanded = box[1] - delta_h / 2
    x_max_expanded = box[2] + delta_w / 2
    y_max_expanded = box[3] + delta_h / 2

    # Return the expanded box
    expanded_box = torch.tensor([[x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded]])
    return expanded_box

def delete_post_background_clusters(result,substrate_result, post_harvest_polygons_info_base,iou_threshold = 0.5):
    result = result.cpu().numpy().to_dict()
    substrate_bbox = expand_box(substrate_result.cpu()["bboxes"][0],0.025)
    to_delete = []

    for idy in range(len(result["pred_instances"]["bboxes"])):
        box2 = torch.tensor([result["pred_instances"]["bboxes"][idy]], dtype=torch.float)
        substrate_iou = bops.box_iou(substrate_bbox, box2)
        ## first check is to have common area with the expanded substrate, this filters out the far away instances
        if substrate_iou==0:

            preds_iou = []
            for idx in range(len(post_harvest_polygons_info_base)):
                box1 = torch.tensor([post_harvest_polygons_info_base[idx][5]], dtype=torch.float)
                preds_iou.append(bops.box_iou(box1, box2))  

            if any(pred_iou>=iou_threshold for pred_iou in preds_iou):
                continue
            else:
                 to_delete.append(idy)
                    
    ## delete from all components of the result variable the overlapping instances with classification/confidence score
    result["pred_instances"]["bboxes"] = np.delete(result["pred_instances"]["bboxes"],to_delete, axis=0)
    result["pred_instances"]["scores"] = np.delete(result["pred_instances"]["scores"],to_delete, axis=0)
    result["pred_instances"]["masks"] = np.delete(result["pred_instances"]["masks"],to_delete, axis=0)
    result["pred_instances"]["labels"] = np.delete(result["pred_instances"]["labels"],to_delete, axis=0)

    return dict_to_det_data_sample(result)

#Calculating intersection over union for coordiante list 
def harvest_filter(polygons,polygons_info,baseline,margin = 0.05,iou_threshold = 0.3):

    #Track which clusters are to be deleted
    to_delete = []

    #Iterating through each pair of polygons and baselines
    i = 0
    for poly in polygons:
        poly1 = Polygon(poly).buffer(0)

        for base in baseline:
            base = base[0]
            poly2 = Polygon(base).buffer(0)

            #Getting intersection of both polygons
            intersect = poly1.intersection(poly2).area
            if intersect == 0:
                continue
            
            #Getting union and intersection over union
            union = poly1.union(poly2).area
            iou = intersect / union

            #Check for full overlap and small cluster (harvested cluster) based on overlap margin
            if (poly1.area >= ((1-margin)*intersect) and poly1.area <= ((1+margin)*intersect) and iou <= iou_threshold):
                to_delete.append(i)
                break        
        
        i += 1

    #Deleting harvested clusters
    for i in reversed(to_delete):
        polygons.pop(i)
        polygons_info.pop(i)

    return polygons, polygons_info, to_delete

    