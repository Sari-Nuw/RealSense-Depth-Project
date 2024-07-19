import os
import mmcv
from mmengine import Config
from mmdet.apis import inference_detector,init_detector
import numpy as np
import cv2
from concave_hull import concave_hull_indexes
from Mask_Generator_Utils_Common import *
from Segment_Crop import *
from Depth_Estimation import *
from DepthMap import *
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from mmdet.registry import VISUALIZERS
from Environmental_Tracking import *
import csv

#Sorting clusters for tracking
def cluster_sort_depth(polygons,polygons_info,polygons_depth_info):

	#Establishing baseline for sorting
	baseline = []

	i = 0
	while baseline == []:
		if polygons[i] != []:
			baseline = polygons[i].copy()
			start = i
		i += 1

    #Sorting the bounding boxes for consistency 
	for i in range(start,len(polygons)-1):
		polygons[i+1],polygons_info[i+1],polygons_depth_info[i+1] = coordinate_sort_depth(polygons[i+1],polygons[i],polygons_info[i+1],polygons_depth_info[i+1],baseline)
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

	return polygons,polygons_info,polygons_depth_info

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
	
#Iterating through the images and performing the predictions and depth estimations
def image_processing_depth(confidence_score_threshold,test_set,test_set_path,predicted_images,averaged_length_pixels,mushroom_model,visualizer,pipe,stereo_option,env_option):

	images = []
	image_files = []
	data = []
	polygons = []
	polygons_info = []
	polygons_depth_info = []
	stereo_depth_images = []
	estimated_depth_images = []
	color_estimated_depth_images = []

	#From the farm substrate (50 cm)
	substrate_real_size = 50

	#Image size in pixels
	img_size = 0

	for test_img in test_set:

		# load the image and color correct
		img = mmcv.imread(test_set_path + test_img)
		img = mmcv.image.bgr2rgb(img)

		#saving image for processing and image file names 
		images.append(img)
		image_files.append(test_set_path + test_img)

		#Read stereo depth map
		if stereo_option:
			break
			# depth_img = cv2.imread(depth_test_images +'/img ({}).png'.format(i))
			# stereo_depth_images.append(DepthMaptoFrame(depth_img,0,2000))

		#ONLY IF ALL IMAGES ARE OF THE SAME SIZE
		#Calculating total number of pixels in the image
		if img_size == 0:
			#To compare with cluster sizes for relative sizing (assuming image has 3 color channels)
			img_size = img.size/3

		#Extracting time data from the images to be used for environmental tracking
		if env_option:
			img_data = Image.open(test_set_path + test_img)._getexif()
			if not img_data:
				data.append('No Time Data')
			else:
				data.append(img_data[36867])

		# Mushroom segmentation inference
		image_result = inference_detector(mushroom_model, img)#       !!!!!!!!!!!!!!
		image_result = delete_low_confidence_predictions(image_result) #				!!!!!!!!!!!!!!
		image_result = delete_overlapping_with_lower_confidence(image_result)	#		!!!!!!!!!!!!!!

		# show the results
		visualizer.add_datasample(
			'result',
			img,
			data_sample=image_result,
			draw_gt = None,
			wait_time=0,
			out_file=predicted_images + "prediction_" + test_img,
			pred_score_thr=confidence_score_threshold
		)

		#Converting how inference information is saved
		img_result = []
		for result in image_result.pred_instances:
			img_result.append([result[0]["scores"][0],result[0]["bboxes"][0],result[0]["labels"][0],result[0]["masks"][0]])

		#To store all the results from the image
		results = []
		results_info = []

		# Converting from a boolean mask to a coordinate mask
		for result in img_result:
			masks = result[3]
			if not np.array_equal(masks,[]):
				#Converting the true/false mask matrix in each result to a coordinate list 
				points = np.argwhere(masks).tolist()
				#Flipping across the coordinates from (y,x) to (x,y)
				points = np.flip(points,1)
				points = points[0::10]
				#Finding the concave hull (outline) of the mask 
				#Note: you can change the concativity of the hull to have a 'tighter' fit
				#A tighter fit can result in errors occuring due to the polygon 'overlapping' itself
				#This can be changed through the library (ctrl+click concave_hull_indexes)
				hull = concave_hull_indexes(points)
				#Appending the points that make the outline
				results.append(points[hull])

				#Caclulating cluster pixel sizing
				pixel_cluster_width = result[1][2] - result[1][0]
				pixel_cluster_height = result[1][3] - result[1][1]
				#Getting cluster label
				cluster_label = result[2]
				## use the last element of the averaged substrate lengths to approximate the actual cluster length and width
				absolute_cluster_width = round(pixel_cluster_width*substrate_real_size/averaged_length_pixels,3)
				absolute_cluster_height = round(pixel_cluster_height*substrate_real_size/averaged_length_pixels,3)
				results_info.append([cluster_label,pixel_cluster_height,pixel_cluster_width,absolute_cluster_height,absolute_cluster_width])
	
		#Saving the hull results for all the clusters in the image
		polygons.append(results)
		polygons_info.append(results_info)

		#Performing monocular depth estimation
		# load again the image through diffuser library
		image: Image.Image = load_image(test_set_path + test_img)
		
		# MDE inference with many configurable variables
		pipeline_output = pipe(
			image,                    # Input image.
			# ----- recommended setting for DDIM version -----
			# denoising_steps=10,     # (optional) Number of denoising steps of each inference pass. Default: 10.
			# ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
			# ------------------------------------------------

			# ----- recommended setting for LCM version ------
			denoising_steps=4,
			ensemble_size=5,
			# -------------------------------------------------

			# processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
			# match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
			# batch_size=0,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
			# seed=100,              # (optional) Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing --batch_size 1 helps to increase reproducibility. To ensure full reproducibility, deterministic mode needs to be used.
			# color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral". Set to `None` to skip colormap generation.
			# show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
		)
		
		# depth variable is an array with 0 - 1 range
		# depth: np.ndarray = pipeline_output.depth_np                    # Predicted depth map
		# depth_colored: Image.Image = pipeline_output.depth_colored      # Colorized prediction
		depth = pipeline_output.depth_np                    # Predicted depth map
		depth_colored = pipeline_output.depth_colored 

		depth_info = []

		for result in img_result:
			if not np.array_equal(result[3],[]): 
				## mask and keep only the depth information of the predicted cluster
				cluster_depth = depth * np.squeeze(result[3])
				## erode the shape to get rid of background areas in the border of the mask
				cluster_depth = mask_erosion(cluster_depth,kernel_size=21,iterations=3)

				depth_info.append([round(np.nanmean(cluster_depth),3),round(np.nanstd(cluster_depth),3),round(np.nanmax(cluster_depth),3),round(np.nanmin(cluster_depth),3)])

		polygons_depth_info.append(depth_info)

		#Saving the estimated depth array and the estimated color depth image
		estimated_depth_images.append(depth)
		color_estimated_depth_images.append(depth_colored)

		# Save depth map to prediction folder 
		depth_uint8 = (depth * 255).astype(np.uint8)
		Image.fromarray(depth_uint8).save(predicted_images  + "prediction_" + test_img[:-4] + "_depth_map.png", mode="I;8")
	

		# Save colorized depth map to prediction folder
		depth_colored.save(predicted_images  + "prediction_" + test_img[:-4] + "_depth_colored.jpg")

	return 	images,image_files,data,polygons,polygons_info,polygons_depth_info,stereo_depth_images,estimated_depth_images,color_estimated_depth_images,img_size
	
#Loading models prediction and depth estimation models
def load_models_depth(configs_folder,mushroom_architecture_selected,substrate_architecture_selected,use_device):
	
	# load the trained model
	mushroom_best_checkpoint_weights = [x for x in os.listdir(configs_folder) if x.endswith("pth") and x.startswith("mushroom")][0]
	substrate_best_checkpoint_weights = [x for x in os.listdir(configs_folder) if x.endswith("pth") and x.startswith("substrate")][0]

	# load the configuration file
	mushroom_architecture_config_file = configs_folder + [x for x in os.listdir(configs_folder) if mushroom_architecture_selected in x and x.endswith(".py")][0]
	substrate_architecture_config_file = configs_folder + [x for x in os.listdir(configs_folder) if substrate_architecture_selected in x and x.endswith(".py")][0]

	# initiate the model
	mushroom_model = init_detector(Config.fromfile(mushroom_architecture_config_file), configs_folder + mushroom_best_checkpoint_weights, device=use_device)
	substrate_model = init_detector(Config.fromfile(substrate_architecture_config_file), configs_folder + substrate_best_checkpoint_weights, device=use_device)

	# init visualizer(run the block only once in jupyter notebook)
	visualizer = VISUALIZERS.build(mushroom_model.cfg.visualizer)
	# the dataset_meta is loaded from the checkpoint and
	# then pass to the model in init_detector
	visualizer.dataset_meta = mushroom_model.dataset_meta
	visualizer.dataset_meta["palette"][0] = (20, 220, 60)
	visualizer.dataset_meta["palette"][1] = (220, 40, 50)

	# Initialize the MDE model
	pipe = DiffusionPipeline.from_pretrained(
		"prs-eth/marigold-v1-0",
		custom_pipeline="marigold_depth_estimation"
		# torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
		# variant="fp16",                           # (optional) Use with `torch_dtype=torch.float16`, to directly load fp16 checkpoint
	)
	pipe.to(use_device)
	
	return mushroom_model,substrate_model,visualizer,pipe

#Isolate the cluster from the original image
def process_cluster_depth(image_copy,depth_img_copy,poly,bounding,working_folder,i,j):

	#Get shape of the image
	height = image_copy.shape[0]
	width = image_copy.shape[1]
	
	#Limiting the upper boundaries to the maximum width and height
	limit_factor = 0.025
	limit_increase = 1 + limit_factor
	limit_decrease = 1 - limit_factor
	if bounding.maxy*limit_increase > height:
		uppery = height
	else:
		uppery = int(bounding.maxy*limit_increase)
	if bounding.maxx*limit_increase > width:
		upperx = width
	else:
		upperx = int(bounding.maxx*limit_increase)
	#Copying the cropped image section
	box_image = image_copy[int(bounding.miny*limit_decrease) : uppery, int(bounding.minx*limit_decrease) : upperx]
	#Getting the box of the matching depth image
	depth_box_image = depth_img_copy[int(bounding.miny*limit_decrease) : uppery, int(bounding.minx*limit_decrease) : upperx]
	#Converting from the polygon coordinates of the full picture to polygon coordinates in the box image
	local_poly = poly.copy()
	local_poly[:,0] = local_poly[:,0] - bounding.minx*limit_decrease
	local_poly[:,1] = local_poly[:,1] - bounding.miny*limit_decrease
	#Saving the bounded section of the image
	cv2.polylines(box_image, np.int32([local_poly]), True, (255, 0, 0), 5)

	#Saving isolated cluster image
	cv2.imwrite(working_folder + "/Cluster/image ({})_cluster ({}).JPG".format(i+1,j), cv2.cvtColor(box_image,cv2.COLOR_RGB2BGR))

	return box_image,depth_box_image,local_poly

#Saving the image information for an individual cluster in numpy array format
def save_cluster_array_depth(sizing_image,poly,centre,box_image,depth_box_image,local_poly,working_folder,i,j):
	cv2.polylines(sizing_image, np.int32([poly]), True, (255, 0, 0), 5)
	cv2.putText(sizing_image, str(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
	#Localizing polygon mask
	binary_mask = np.zeros((box_image.shape[0],box_image.shape[1]),int)
	for point in local_poly:
		binary_mask[point[1],point[0]] = 1
	array = [box_image[:,:,0],box_image[:,:,1],box_image[:,:,2],depth_box_image,binary_mask]
	np.save(working_folder + "/Arrays/RGBD_image ({})_cluster ({})".format(i+1,j),array)

#Saving the various image types
def save_image_depth(working_folder,full_image,sizing_image,estimated_depth_images,color_estimated_depth_images,cluster_sizing_option,i):
	cv2.imwrite(working_folder + "/Picture/images ({}).JPG".format(i+1), cv2.cvtColor(full_image,cv2.COLOR_RGB2BGR))
	if cluster_sizing_option:
		cv2.imwrite(working_folder + "/Sizing/images ({}).JPG".format(i+1), cv2.cvtColor(sizing_image,cv2.COLOR_RGB2BGR)) 
	depth_image = DepthMap(estimated_depth_images[i],0,1)
	cv2.imwrite(working_folder + "/Depth/predicted__depth_image ({}).JPG".format(i+1), np.array(depth_image))
	cv2.imwrite(working_folder + "/Depth/predicted_color_depth_image ({}).JPG".format(i+1), cv2.cvtColor(np.array(color_estimated_depth_images[i]),cv2.COLOR_RGB2BGR))

#Saving the image information in numpy array format
def save_image_array_depth(full_image,depth_img_copy,polygon,working_folder,i):
	array = [full_image[:,:,0],full_image[:,:,1],full_image[:,:,2],depth_img_copy]
	for poly in polygon:
		binary_mask = np.zeros((full_image.shape[0],full_image.shape[1]),int)
		for point in poly:
			if not isinstance(point,int):
				binary_mask[point[1],point[0]] = 1
		array.append(binary_mask)
	np.save(working_folder + "/Arrays/RGBD_image ({})".format(i+1),array)

#Writing information from cluster_segments to excel file
def write_cluster_sizing_depth(cluster_segments,working_folder):
    with open(working_folder + '/Cluster_Sizing.csv', 'w',newline='') as csv_file:
        #Creating the csv writer
        writer = csv.writer(csv_file)
        #Writing the first row with all the headers
        writer.writerow(['Image #','Cluster #','Cluster Area','Label','Cluster Height','Cluster Width','Absolute Height','Absolute Width','Vetical Left','Vertical Middle','Vertical Right','Horizontal Top','Horizontal Middle','Horizontal Bottom','Depth Mean','Depth Std Dev','Depth Max','Depth Min'])
        for segment in cluster_segments:
            if segment == []:
                writer.writerow('')
            else:
                writer.writerow(["img ({})".format(segment[0]),segment[1],segment[2],segment[3][0],segment[3][1],segment[3][2],segment[3][3],segment[3][4],segment[4][0],segment[4][1],segment[4][2],segment[4][3],segment[4][4],segment[4][5],segment[5][0],segment[5][1],segment[5][2],segment[5][3]])
