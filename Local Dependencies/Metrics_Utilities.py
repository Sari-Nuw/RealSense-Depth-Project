import csv
import json
import numpy as np
from Sorting_utilities import coordinate_iou
import torch
import torchvision.ops.boxes as bops

## converts COCO to VOC annotation format
def COCO_to_VOC_bbox(bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    return [x1,y1,x2,y2]

def annotation_tracking(file):
	
    ## set input and output paths
    tracked_annotations_save_path = "{}_tracked.json".format(file.replace('.json',''))
    ## open input annotation file
    with open(file) as f:
        annotations = json.load(f)


    ## initialize variable for baseline bboxes
    baseline_bboxes = []
    for idx,annotation in enumerate(annotations["annotations"]):
        ## Convert COCO x,y,h,w to VOC x1,y1,x2,y2 for correct results in the similarity function
        converted_bbox = COCO_to_VOC_bbox(annotation["bbox"])
        ## baseline_bboxes is not empty
        if baseline_bboxes:
            ## calculate cosine similarity between the new bbox and all existing baseline bboxes
            # cosine_similarity = np.dot(baseline_bboxes,converted_bbox)/(norm(baseline_bboxes, axis=1)*norm(converted_bbox)) #np.round(, 4)
            cosine_similarity = []
            for baseline_bbox in baseline_bboxes:
                cosine_similarity.append(bops.box_iou(torch.tensor([converted_bbox], dtype=torch.float), torch.tensor([baseline_bbox], dtype=torch.float))[0][0].item() )
            ## if no baseline bbox has high cosine similarity with the new bbox then we have a new instance and we add the new bbox to the baseline
            ## the threshold of 0.998 was found after multiple experimental runs and visualizations of the baseline bboxes to detect all real clusters with no duplicates
            if (np.array(cosine_similarity)<0.4).all():
                ## add the new bbox to the baseline
                baseline_bboxes.append(converted_bbox)
                ## add the tracking_id field to the annotation variable, which is the found index position
                annotation["tracking_id"] = len(baseline_bboxes)-1
            ## else if there is one baseline bbox with high cosine similarity with the new bbox then we have an instance match
            else:
                ## numpy array to list for code simplicity
                # cosine_similarity = cosine_similarity.tolist()
                ## find the index of the baseline bbox that matches, cosine_similarity and baseline_bboxes have the same order
                respective_baseline_bbox_index = cosine_similarity.index(max(cosine_similarity))
                ## update the index position of the baseline_bboxes variable with the new bbox
                baseline_bboxes[respective_baseline_bbox_index] = converted_bbox
                ## add the tracking_id field to the annotation variable, which is the found index position
                annotation["tracking_id"] = respective_baseline_bbox_index
        ## if baseline_bboxes is empty add the first element to initialize the process
        else:
            ## add the first bbox to the baseline
            baseline_bboxes.append(converted_bbox)
            ## add the tracking_id field to the annotation variable, which is the found index position
            annotation["tracking_id"] = 0

    # save the new tracked annotations
    with open(tracked_annotations_save_path,"w") as f:
        json.dump(annotations,f)

    # Read the new json file
    annotations, sorting_annotations = get_annotations_json(tracked_annotations_save_path)
	
    return annotations, sorting_annotations

#Establishing cluster_segments excel file
def establish_cluster_sizing(working_folder):
	with open(working_folder + '/Cluster_Sizing.csv', 'w') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file)
		#Writing the first row with all the headers
		writer.writerow(['Image #','Cluster #','Cluster Pixel Area','Absolute Cluster Area','Label','Cluster Height','Cluster Width','Absolute Height','Absolute Width','Vetical Left','Vertical Middle','Vertical Right','Horizontal Top','Horizontal Middle','Horizontal Bottom'])

#Writing information from precision_metrics to excel file
def establish_metrics(working_folder):
	with open(working_folder + '/Precision_Metrics.csv', 'w') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file)
		#Writing the first row with all the headers
		writer.writerow(['Image #','Metric Type','mAP','mAR','AP50','AP75','F1','True Positive [0.5:0.95]','False Positive [0.5:0.95]','False Negative [0.5:0.95]'])

# Establishing csv file for multiple object tracking accuracy
def establish_mota(working_folder):

	with open(working_folder + '/MOTA_Metrics.csv', 'w') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file)
		#Writing the first row with all the headers
		writer.writerow(['Image #','MOTA 50','FP 50','Cumulative FP 50','FN 50','Cumulative FN 50','IDS 50', 'Cumulative IDS 50','GT 50','Cumulative GT 50','MOTA 75','FP 75','Cumulative FP 75','FN 75','Cumulative FN 75','IDS 75', 'Cumulative IDS 75','GT 75','Cumulative GT 75'])

#Getting annotations from text file (JSON Coco format)
def get_annotations_json(text_file):

	#Pathway to json file
	with open(text_file, 'r') as file:
		# Reading from json file
		annotation_data = json.load(file)

	# Annotations for each image
	annotations = [[] for _ in range(len(annotation_data['images']))]
	sorting_annotations = [[] for _ in range(len(annotation_data['images']))]
	
	for annotation in annotation_data['annotations']:

		# Pairing each pair of x,y points together
		points = annotation['segmentation'][0]
		segment = []
		pair = []
		for point in points:
			pair.append(int(point))
			if len(pair) == 2:
				segment.append(pair)
				pair = []
		
		# Full polygon added to corresponding image
		annotations[annotation['image_id']-1].append([segment,annotation['category_id']])
		sorting_annotations[annotation['image_id']-1].append([segment,annotation['tracking_id'],annotation['category_id']])

	return annotations, sorting_annotations

#Process annotation metrices for each image
def get_annotation_metrics(annotations,polygons,polygons_info,TP_array, FP_array, FN_array):

    # Removing placeholder [0] from list for metrics
    polygon_metrics = []
    polygon_metrics_info = []
    i = 0
    for poly in polygons:
        if len(poly) > 1:
            polygon_metrics.append(poly)
            polygon_metrics_info.append(polygons_info[i])
        i += 1

    #Storing true positive, false positive, and false negative values at different iou threshold values
    TP_array_current = []
    FP_array_current = []
    FN_array_current = []

    #Iterating between iou_threshold of 0.5-0.95 to calculate the mAP,mAR, and F1-score
    iou_threshold = 0.5
    k = 0
    while iou_threshold < 1:
        #Track true positive detections
        TP = 0
        #To check whether an annotation/polygon have been recognized/matched 
        annotations_check = [False for _ in range(len(annotations))]
        j = 0
        for polygon in polygon_metrics:
            #To iterate across annotation check
            i = 0
            for annotation in annotations:
                iou = coordinate_iou(annotation[0],polygon)
                if iou >= iou_threshold:
                    print('here')
                    #Each annotation should only be detected once. Extra detections ae false positives
                    #Label condition also being checked
                    if annotations_check[i] == False:
                         print('check1')
                    if annotations_check[i] == False and annotation[1] == (polygon_metrics_info[j][0]+1):
                        print('check2')
                        TP += 1
                        annotations_check[i] = True
                i += 1
            j += 1

        #False Negative. Annotation was never detected
        FN = len(annotations) - TP

        #False positive. Polygon was detected where there should not be a detection
        FP = len(polygon_metrics) - TP

        #Saving the metrics at this interval
        TP_array_current.append(TP)
        FP_array_current.append(FP)
        FN_array_current.append(FN)

        iou_threshold += 0.05
        k += 1

    TP_array.append(TP_array_current)
    FP_array.append(FP_array_current)
    FN_array.append(FN_array_current)

    if k == 0:
        AP50 = TP/(TP+FP)
    elif k == 5:
        AP75 = TP/(TP+FP)

    #Cumulative across all previous images
    cum_TP = [0 for _ in range(10)]
    cum_FP = [0 for _ in range(10)]
    cum_FN = [0 for _ in range(10)]
    for i in range(10):
        for j in range(len(TP_array)):
             cum_TP[i] = cum_TP[i] + TP_array[j][i]
             cum_FP[i] = cum_FP[i] + FP_array[j][i] 
             cum_FN[i] = cum_FN[i] + FN_array[j][i]

    # print(TP_array)
    # print(FP_array)
    # print(FN_array)
    # print(cum_TP)
    # print(cum_FP)
    # print(cum_FN)

    #Claculating metrics
    AP50 = cum_TP[0]/(cum_TP[0]+cum_FP[0])
    AP75 = cum_TP[5]/(cum_TP[5]+cum_FP[5])
    mAP = sum(cum_TP)/(sum(cum_TP)+sum(cum_FP))
    mAR = sum(cum_TP)/(sum(cum_TP)+sum(cum_FN))
    if mAP == 0 and mAR == 0:
         F1 == 0
    else:
        F1 = (2*mAP*mAR)/(mAP+mAR)

    return mAP, mAR, AP50, AP75, F1, TP_array, FP_array, FN_array

def get_sorting_metrics(annotations,polygons,mota_metrics_50,mota_metrics_75, motaTracker_50, motaTracker_75):

    GT = len(annotations)
    mota_metrics_50[3].append(GT)
    mota_metrics_75[3].append(GT)

    max_id = -1
    for annotation in annotations:
            if annotation[1] > max_id:
                max_id = annotation[1]
	
    i = 0
    while i < 2:
        if i == 0:
            iou_threshold = 0.5
        elif i ==1:
            iou_threshold = 0.75
        mota_id = [[] for _ in range(max(len(polygons),len(annotations)))]
        # Track how many of the polygons have corresponding ground truth annotations
        polygon_check = 0
        FN = 0
        IDS = 0
        for annotation in annotations:

            tracking_id = annotation[1]
            annotation = annotation[0]			
            # To check if an annotation has been tracked
            annotation_tracked = False
            max_iou = 0
            index = 0
            max_index = -1
            for polygon in polygons:

                if len(polygon) > 1:

                    iou = coordinate_iou(annotation,polygon)
                
                    if iou > iou_threshold:
                        # Increment polygon check (max once per annotation)
                        if not annotation_tracked:
                            polygon_check += 1
                        # Annotation has been tracked
                        annotation_tracked = True
                        # In case of overlapping annotations
                        if iou > max_iou:
                            max_iou = iou
                            max_index = index

                index += 1
            
            if max_index != -1:
                mota_id[max_index].append(tracking_id)
                #Check that tracker has been established
                if i == 0:
                    if len(motaTracker_50) > 0:
                        #Check that tracker isn't being compared to a new cluster
                        if len(motaTracker_50[-1]) > max_index: 
                            print('check50')
                            print(motaTracker_50[-1])
                            print(max_index)
                            if len(motaTracker_50[-1][max_index]) > 0 and motaTracker_50[-1][max_index][0] != tracking_id:
                                print('id check')
                                print(motaTracker_50[-1][max_index], tracking_id)
                                IDS += 1
                elif i == 1:
                    if len(motaTracker_75) > 0:
                        #Check that tracker isn't being compared to a new cluster
                        if len(motaTracker_75[-1]) > max_index: 
                            print('check75')
                            print(motaTracker_75[-1])
                            print(max_index)
                            if len(motaTracker_75[-1][max_index]) > 0 and motaTracker_75[-1][max_index][0] != tracking_id:
                                print('id check')
                                print(motaTracker_75[-1][max_index], tracking_id)
                                IDS += 1

            # Annotation has not been tracked	
            if not annotation_tracked:
                FN += 1

        polygon_count = 0
        for polygon in polygons:
            if len(polygon) > 1:
                polygon_count += 1

        FP = polygon_count - polygon_check

        if i == 0:
            mota_metrics_50[0].append(FP)
            mota_metrics_50[1].append(FN)
            mota_metrics_50[2].append(IDS)
            motaTracker_50.append(mota_id)
        elif i == 1:
            mota_metrics_75[0].append(FP)
            mota_metrics_75[1].append(FN)
            mota_metrics_75[2].append(IDS)
            motaTracker_75.append(mota_id)
        
        i += 1

    return mota_metrics_50, mota_metrics_75, motaTracker_50, motaTracker_75

#Writing information from cluster_segments to excel file
def write_cluster_sizing(segment,working_folder):
	with open(working_folder + '/Cluster_Sizing.csv', 'a') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file,lineterminator='\n')
		#Writing new row
		if segment == []:
			writer.writerow('\n')
		else:
			writer.writerow(["img ({})".format(segment[0]),segment[1],segment[2],segment[3],segment[4][0],segment[4][1],segment[4][2],segment[4][3],segment[4][4],segment[5][0],segment[5][1],segment[5][2],segment[5][3],segment[5][4],segment[5][5]])

#Writing information from cluster_segments to excel file
def write_metrics(working_folder,metric_type,metrics,img_num):
	with open(working_folder + '/Precision_Metrics.csv', 'a') as csv_file:
		#Creating the csv writer
		writer = csv.writer(csv_file,lineterminator='\n')
		#Writing new row
		writer.writerow(["img ({})".format(img_num+1),metric_type,round(metrics[0],2),round(metrics[1],2),round(metrics[2],2), round(metrics[3],2),round(metrics[4],2),metrics[5],metrics[6],metrics[7]])

#Writing information from cluster_segments to excel file
def write_mota(working_folder,mota_metrics_50, mota_metrics_75,img_num):

    cum_FP_50 = sum(mota_metrics_50[0])
    cum_FN_50 = sum(mota_metrics_50[1])
    cum_IDS_50 = sum(mota_metrics_50[2])
    cum_GT_50 = sum(mota_metrics_50[3])

    mota_50 = 1 - ((cum_FP_50 + cum_FN_50 + cum_IDS_50)/cum_GT_50)
        
    cum_FP_75 = sum(mota_metrics_75[0])
    cum_FN_75 = sum(mota_metrics_75[1])
    cum_IDS_75 = sum(mota_metrics_75[2])
    cum_GT_75 = sum(mota_metrics_75[3])

    mota_75 = 1 - ((cum_FP_75 + cum_FN_75 + cum_IDS_75)/cum_GT_75)

    with open(working_folder + '/MOTA_Metrics.csv', 'a') as csv_file:
        #Creating the csv writer
        writer = csv.writer(csv_file,lineterminator='\n')
        #Writing new row
        writer.writerow(["img ({})".format(img_num+1),mota_50,round(mota_metrics_50[0][-1],2),cum_FP_50,mota_metrics_50[1][-1],cum_FN_50,mota_metrics_50[2][-1],cum_IDS_50,mota_metrics_50[3][-1],cum_GT_50,mota_75,round(mota_metrics_75[0][-1],2),cum_FP_75,mota_metrics_75[1][-1],cum_FN_75,mota_metrics_75[2][-1],cum_IDS_75,mota_metrics_75[3][-1],cum_GT_75])