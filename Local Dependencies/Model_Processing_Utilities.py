from concave_hull import concave_hull_indexes
import cv2
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmengine import Config
import numpy as np
import os
import torch 

from Filtration_Utilities import expand_box

#Checking if CUDA is available on the device running the program
def check_cuda():
	if torch.cuda.is_available():
		use_device = "cuda"
	else:
		use_device = "cpu"
	print(use_device)
	return use_device

#Loading models prediction
def load_models(configs_folder,mushroom_architecture_selected,substrate_architecture_selected,use_device):
	
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

	return mushroom_model,substrate_model,visualizer

def process_substrate_results(img,substrate_result,working_folder,img_num):

	substrate_img = img.copy()

	#Draw bounding boxes on substrate images
	for result in substrate_result:
		sub_result = result["bboxes"].cpu().numpy()[0]
		cv2.rectangle(substrate_img,(int(sub_result[0]),int(sub_result[1])),(int(sub_result[2]),int(sub_result[3])),(0,0,255),5)

	#Save substrate images
	cv2.imwrite(working_folder + "/Substrate/images ({}).JPG".format(img_num+1), cv2.cvtColor(substrate_img,cv2.COLOR_RGB2BGR))

#Processing image polygons and information
def process_results(image_result,averaged_length_pixels, substrate_real_size = 50):

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
			#Reducing the size of the coordinate list
			points = points[0::10]
			#Finding the concave hull (outline) of the mask 
			hull = concave_hull_indexes(points)
			#Appending the points that make the outline
			results.append(points[hull])

			#Caclulating cluster pixel sizing
			cluster_bbox = np.array([result[1][0],result[1][1],result[1][2],result[1][3]]).astype(int)
			pixel_cluster_width = result[1][2] - result[1][0]
			pixel_cluster_height = result[1][3] - result[1][1]

			#Getting cluster label
			cluster_label = result[2]

			## use the last element of the averaged substrate lengths to approximate the actual cluster length and width
			absolute_cluster_width = round(pixel_cluster_width*substrate_real_size/averaged_length_pixels[-1],3)
			absolute_cluster_height = round(pixel_cluster_height*substrate_real_size/averaged_length_pixels[-1],3)

			#Results info include: Cluster label, bbox height/width, absolute height/width, bbox coordinates 
			results_info.append([cluster_label,pixel_cluster_height,pixel_cluster_width,absolute_cluster_height,absolute_cluster_width,cluster_bbox])

	return results, results_info