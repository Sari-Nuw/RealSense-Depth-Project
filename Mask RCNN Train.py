import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.structures.instance_data import InstanceData

from matplotlib import pyplot as plt
import cv2 
import torch
import torchvision.ops.boxes as bops
import copy
import numpy as np
import os
import time
from natsort import natsorted

import random

# Check Pytorch installation
import torch, torchvision
print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print("mmdetection:",mmdet.__version__)

# Check mmcv installation
import mmcv
print("mmcv:",mmcv.__version__)

# Check mmengine installation
import mmengine
print("mmengine:",mmengine.__version__)

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

def delete_overlapping_with_lower_confidence(result,iou_threshold = 0.9):
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


project_folder = "C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Mask RCNN Training/"
## Dataset settings
path_to_dataset = project_folder + "Full1/"
path_to_test_set = project_folder + "Partial1"

## Architecture settings
architecture_selected = "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco" #_14_28_4
architectures_config_paths_dict = {
    "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco" : project_folder + "configs/mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco"
}

variable_to_find_py_config_file = architecture_selected
architecture_selected += "/"

## Folder structure creation
working_folder = project_folder + "/results/" + architecture_selected
online_pretrained_weights_folder = project_folder + "configs/"
training_outputs = working_folder + "training_outputs/"
test_outputs = working_folder + "test_outputs/"
predicted_images = test_outputs + "predicted_images/"
predicted_numerical_data = test_outputs + "predicted_numerical_data/"
configs_folder = working_folder + "configs_output/"

## Check whether the specified path exists or not
os.makedirs(working_folder,exist_ok=True)
os.makedirs(training_outputs,exist_ok=True)
os.makedirs(configs_folder,exist_ok=True)
os.makedirs(test_outputs,exist_ok=True)
# os.makedirs(predicted_images,exist_ok=True)
# os.makedirs(predicted_numerical_data,exist_ok=True)

architecture_config_file = architectures_config_paths_dict[variable_to_find_py_config_file]
files = os.listdir(online_pretrained_weights_folder)
architecture_pretrained_file = online_pretrained_weights_folder + [x for x in files if x.startswith(variable_to_find_py_config_file) and x.endswith(".pth")][0]

dataset_to_be_used = "small_timelapse"
use_online_pretrained = True
config_save_filename = "custom_config_" + architecture_selected[:-1]
training_validation = True
prediction_images_threshold = 0.5
save_pickle_files = False
if save_pickle_files:
    confidence_filtering_threshold = 0.5

if use_online_pretrained:
    print("Using official pretrained model configuration and weights with transfer learning")
    print("--------------------------------------------------------------------------------")
    #select model config file and pretrained data
    #cfg = Config.fromfile(architecture_config_file)
    cfg = Config.fromfile('C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Mask RCNN Training/configs/mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py')
    cfg.load_from = architecture_pretrained_file

    # Modify dataset type and path
    cfg.dataset_type = 'COCODataset'

    ##----------------------------------------------------------------------------------------------------------------------------------------
    if dataset_to_be_used=="small_timelapse":
        
        cfg.metainfo = {
            'classes': ('Mushrooms'),
            'palette': [
                    (220, 40, 50)
                ] 
            }

        cfg.data_root = path_to_dataset
        
        ## Timelapse train set
        cfg.train_dataloader.dataset.ann_file = 'Germany_Images.json'
        cfg.train_dataloader.dataset.data_prefix.img = 'Germany_Images/'
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.train_dataloader.dataset.metainfo = cfg.metainfo
        
        ## Timelapse validation set
        cfg.val_dataloader.dataset.ann_file = 'Germany_Val_Images.json'
        cfg.val_dataloader.dataset.data_prefix.img = 'Germany_Val_Images/'
        cfg.val_dataloader.dataset.metainfo = cfg.metainfo
        cfg.val_dataloader.dataset.data_root = cfg.data_root
        cfg.val_evaluator.ann_file = cfg.data_root + cfg.val_dataloader.dataset.ann_file
        cfg.val_evaluator.classwise = True
        
        ## Timelapse test set
        cfg.test_dataloader.dataset.ann_file = 'Germany_Val_Images.json'
        cfg.test_dataloader.dataset.data_prefix.img = 'Germany_Val_Images/'
        cfg.test_dataloader.dataset.metainfo = cfg.metainfo  
        cfg.test_dataloader.dataset.data_root = path_to_test_set
        cfg.test_evaluator.ann_file = path_to_test_set + cfg.test_dataloader.dataset.ann_file
        cfg.test_evaluator.classwise = True
        
#
        # ## If no test-set is present
        # cfg.test_dataloader.dataset.ann_file = cfg.val_dataloader.dataset.ann_file 
        # cfg.test_dataloader.dataset.data_prefix.img = cfg.val_dataloader.dataset.data_prefix.img 
        # cfg.test_dataloader.dataset.metainfo = cfg.val_dataloader.dataset.metainfo
        # cfg.test_dataloader.dataset.data_root = cfg.val_dataloader.dataset.data_root
        # cfg.test_evaluator.ann_file = cfg.val_evaluator.ann_file

    else:
        pass
        ### Placeholder for a different future dataset 
            

    ##----------------------------------------------------------------------------------------------------------------------------------------
    
    ### modify number of images per batch
    cfg.train_dataloader.batch_size = 4
    cfg.val_dataloader.batch_size = 4
    cfg.test_dataloader.batch_size = 4
    
    ### keep it 1 for reproducability, increase if enough hardware and no need to reproduce
    cfg.train_dataloader.num_workers = 1
    cfg.val_dataloader.num_workers = 1
    cfg.test_dataloader.num_workers = 1
    
    # Set up working dir to save files and logs.
    # cfg.work_dir = './tutorial_exps'
    cfg.work_dir = training_outputs

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU, if needed chnage it further.
    total_epochs = 20
    cfg["optim_wrapper"]["optimizer"]["lr"] = 0.0001 #0.02 / 8
    cfg["param_scheduler"][1]["end"] = total_epochs
    cfg["param_scheduler"][1]["milestones"] = [total_epochs]
    
    
    cfg["default_hooks"]["logger"]["interval"] = total_epochs
    
    cfg["default_hooks"]["checkpoint"]["save_best"] = "coco/segm_mAP"
    cfg["default_hooks"]["checkpoint"]["rule"] = "greater"
    del cfg["default_hooks"]["checkpoint"]["interval"]
    
    cfg["train_cfg"]["max_epochs"] = total_epochs
    cfg["train_cfg"]["val_interval"] = 2
    
    #### experiment with micro-changes in Mask R-CNN, such as the branch CNN
    # cfg["model"]["roi_head"]["mask_roi_extractor"]["roi_layer"]["output_size"] = 42
    # cfg["model"]["train_cfg"]["rcnn"]["mask_size"] = 84
    # cfg["model"]["roi_head"]["mask_roi_extractor"]["roi_layer"]["output_size"] = 28
    # cfg["model"]["train_cfg"]["rcnn"]["mask_size"] = 56
    cfg["model"]["roi_head"]["mask_roi_extractor"]["roi_layer"]["output_size"] = 14
    cfg["model"]["train_cfg"]["rcnn"]["mask_size"] = 28
    # cfg["model"]["roi_head"]["mask_head"]["num_convs"] = 6
    
    
    cfg["default_hooks"]["visualization"]["score_thr"] = 0.5

    # Set the seed for all relevant libraries for reproducibility
    seed = 1
    cfg.seed = seed
    set_random_seed(seed, deterministic=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg.gpu_ids = range(1)

    
    del cfg["param_scheduler"][0]


    print(cfg.keys())
    print("-------------------")
    # cfg["device"] = 'cpu'
    cfg["device"] = 'cuda'
    # cuda
    print(cfg.keys())
    print("-------------------")
    

    #for timelapse define resize to match the initial ratio
    resize_w = 1066
    resize_h = 600
    # MAKE COMPATIBLE WITH THE 3-CLASS PROBLEM WE FACE
    ## FOR MASK R-CNN 
    cfg.model.roi_head.bbox_head.num_classes = 3
    cfg.model.roi_head.mask_head.num_classes = 3
    if architecture_selected.endswith("3x_coco/"):
        # cfg.fp16 = dict(loss_scale=512.)
        # cfg.runner.meta = dict(fp16=cfg.fp16)
        pass
    else:
        cfg["train_pipeline"][2]["scale"] = (resize_w,resize_h)
        cfg["test_pipeline"][1]["img_scale"] = (resize_w,resize_h)
        cfg["train_dataloader"]["dataset"]["pipeline"][2]["scale"] = (resize_w,resize_h)
        cfg["val_dataloader"]["dataset"]["pipeline"][1]["scale"] = (resize_w,resize_h)
        cfg["test_dataloader"]["dataset"]["pipeline"][1]["scale"] = (resize_w,resize_h)   
    
      
    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

   
    # SAVE CONFIG FILE TO .PY FILE
    text_file = open(configs_folder + config_save_filename + ".py", "w")
    text_file.write(cfg.dump())
    text_file.close()
      
    
else:    
    # LOAD CONFIG FILE FROM CUSTOM SAVED .PY FILE AND CHANGE THE PRETRAINED WEIGHT PATH
    architecture_config_file = configs_folder + [x for x in os.listdir(configs_folder) if variable_to_find_py_config_file in x and x.endswith(".py")][0]
    architecture_pretrained_file = training_outputs + [x for x in os.listdir(training_outputs) if x.startswith("best")][0]
    # cfg.work_dir = working_folder
    cfg = Config.fromfile(architecture_config_file)
    cfg["load_from"] = architecture_pretrained_file

    print("Using custom pretrained model configuration file: ", architecture_config_file)
    print("Using custom pretrained model weights: ", architecture_pretrained_file)
    print("---------------------------------------------------------")    
    
    # print(f'Config:\n{cfg.pretty_text}')

path_to_train_py = "/home/scio/mmdetection/tools/train.py"
path_to_config_file = configs_folder + config_save_filename + ".py"
#get_ipython().system('python {path_to_train_py} {path_to_config_file}')


## Create and execute the training command
path_to_test_py = "/home/scio/mmdetection/tools/test.py"
path_to_config_file = configs_folder + config_save_filename + ".py"
path_to_best_checkpoint = training_outputs + [x for x in os.listdir(training_outputs) if x.startswith("best")][0]
path_to_images_output_folder = test_outputs + "test_predictions"
path_to_outputs = test_outputs
path_to_output_txt = path_to_outputs + "output.txt"
path_to_pkl_file = path_to_outputs + "results.pkl"

## With and without --show-dir {path_to_images_output_folder} in order to save or not the predicted images as files
# !python {path_to_test_py} {path_to_config_file} {path_to_best_checkpoint} --work-dir {path_to_outputs} --out {path_to_pkl_file} > {path_to_output_txt}
#NOTE: get_ipython().system('python {path_to_test_py} {path_to_config_file} {path_to_best_checkpoint} --work-dir {path_to_outputs} --out {path_to_pkl_file} --show-dir {path_to_images_output_folder} > {path_to_output_txt}')


## Create and execute the confusion matrix command
path_to_conf_matrix_py = "/home/scio/mmdetection/tools/analysis_tools/confusion_matrix.py"
#NOTE: get_ipython().system('python {path_to_conf_matrix_py} {path_to_config_file}  {path_to_pkl_file}  {path_to_outputs} --show --color-theme coolwarm')


## Create and execute the LOSSES curves command, it is necessery to point to the correct json file that has the training logs
path_to_logs_py = "/home/scio/mmdetection/tools/analysis_tools/analyze_logs.py"
path_to_logs = "/home/scio/christos/Mushrooms/cluster_mushrooms/results/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/training_outputs/20241104_122931/vis_data/20241104_122931.json"
path_to_save_losses = path_to_outputs + "losses.png"
#NOTE: get_ipython().system('python {path_to_logs_py} plot_curve {path_to_logs} --keys loss_cls loss_bbox loss_mask --legend loss_cls loss_bbox loss_mask --out {path_to_save_losses}')

## Create and execute the ACCURACY curve command, it is necessery to point to the correct json file that has the training logs
path_to_logs_py = "/home/scio/mmdetection/tools/analysis_tools/analyze_logs.py"
path_to_logs = "/home/scio/christos/Mushrooms/cluster_mushrooms/results/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/training_outputs/20241104_122931/vis_data/20241104_122931.json"
path_to_save_accuraccy = path_to_outputs + "accuraccy.png"
#NOTE: get_ipython().system('python {path_to_logs_py} plot_curve {path_to_logs} --keys acc --legend acc  --out {path_to_save_accuraccy}')

