import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from MonocularDepthEstimation import *
from DepthMap import *
import time
from Segment_Crop import *

start = time.time()

sys.path.append("..")

#Reading image and converting to RGB from BGR
image = cv2.imread(r'Pictures\Picture Date 27-07-2023 Time 17_26_07.png')

#Converting image to RGB from BGR (cv2 reads as Blue Green Red)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Show original image
plt.imshow(image)
plt.title('Original')
plt.show()

# #Loading the segment anything models from files
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# #Set to use GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# #Loading the segment anything models
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#Tunable parameters affect the segmentation of the image
#Link below discuesses the parameters:
#https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35

#Mask generator to crop and segment images (with different parameters)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=24,
    points_per_batch=128,
    pred_iou_thresh=0.99,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

#Generating image mask based on image
masks = mask_generator.generate(image)

#Generating and showing the different image segments
for mask in masks:
        segment = Segment_Crop(mask['segmentation'], plt.gca(), image)
        plt.imshow(segment)
        plt.title('Segment')
        plt.show()



    


