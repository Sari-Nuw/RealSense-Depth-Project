#Take images and segemnt each mushroom (possible issues with tracking the same mushroom growth because no segemnt names)
#Find the number of pixels in the segemnt
#Find the 'area' of the mushrooms as well
#Track the change in pixel number / 'area compared to date and time
#Graph the changes with time and compare with temperature and humidty (environmental conditions) as well

import os
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from MonocularDepthEstimation import *
from DepthMap import *
from ultralytics import YOLO
from Segment_Crop import *
from Volume_Estimation import *

#Path to picture folder
path = r'C:\Users\nuway\OneDrive\Desktop\Realsense Project\Workshop Trial\Picture'

#To open and store the images
images = []
depth_maps = []
bounded_objects = []
i = 1 

#Opening the image and depth images (original image names changed from time to numebring to open)
for file in os.listdir(path):
    image = cv2.imread(r'C:\Users\nuway\OneDrive\Desktop\Realsense Project\Workshop Trial\Picture\Picture ({}).png'.format(i))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    bounded_objects.append(image.copy())
    depth_map = cv2.imread(r'C:\Users\nuway\OneDrive\Desktop\Realsense Project\Workshop Trial\Estimated Depth Map\Estimated Depth Map ({}).png'.format(i))
    depth_maps.append(depth_map)
    i += 1

print(len(images))
print(len(depth_maps))

#Path to object detection model
model_path = r'runs\detect\train7\weights\last.pt'

# Load a model
model = YOLO(model_path)

#Setting threshold for object detection 
threshold = 0.4

#Getting results of object detection
results = []
for image in images:    
    results.append(model(image)[0])

#Pixel coordinates of objects
box_points = [[] for _ in range(len(results))]

objects = 0

#Saving the position of the object boxes
for i in range(len(results)):
    for result in results[i].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            objects = objects + 1
            points = [int(x1),int(y1),int(x2),int(y2)]
            box_points[i].append(points)
            #To see bounding box around detected object
            cv2.rectangle(bounded_objects[i], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(bounded_objects[i], results[i].names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#Printing object pixel coordinates
# for i in range(len(box_points)):   
#     print(i)
#     print(box_points[i])
print(objects)

#To check if any objects were detected
if box_points == []:
    available_box = False
else:
    available_box = True

for i in range(len(images)):
    #Show original image
    # plt.imshow(images[i])
    # plt.title('Original {}'.format(i))
    # plt.show()

    #Show image with bounding boxes
    plt.imshow(bounded_objects[i])
    plt.title('Bounded Objects {}'.format(i))
    plt.show()

# #Loading the segment anything models from files
# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# #Set to use GPU device if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# #Loading the segment anything models
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# #Generating image masks based on box prompts 
# if available_box == True:
#     predictor = SamPredictor(sam)
#     predictor.set_image(image)
#     input_boxes = torch.tensor(box_points, device=predictor.device)
#     transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
#     masks, _, _ = predictor.predict_torch(
#         point_coords=None,
#         point_labels=None,
#         boxes=transformed_boxes,
#         multimask_output=False,
#     )

# #Finding average depth
# if available_box == True:
#     #Calculate depth frame values from depth map colors and minimum/maximum distance (in mm)
#     estimated_depth_frame = DepthMaptoFrame(estimated_depth_map, min_dist, max_dist)
#     if stereo_available:
#         stereo_depth_frame = DepthMaptoFrame(stereo_depth_map, min_dist, max_dist)
#     # np.savetxt('depthdiff',(depth_frame-depth1))
#     # print((depth_frame-depth1).max())
#     # print((depth_frame-depth1).min())
#     #Cropping the depth frame values according to the mask
#     estimated_masked_depth = Segment_Crop(masks[0].cpu().numpy(), plt.gca(), estimated_depth_frame, depth = True)
#     if stereo_available:
#         stereo_masked_depth = Segment_Crop(masks[0].cpu().numpy(), plt.gca(), stereo_depth_frame, depth = True)
#     #Average depth calculated
#     estimated_average_depth = Depth_Calculation(estimated_masked_depth)
#     if stereo_available:
#         stereo_average_depth = Depth_Calculation(stereo_masked_depth)
#     print("Estimated Average Depth (mm2): " + str(estimated_average_depth))
#     if stereo_available:
#         print("Stereo Average Depth (mm2): " + str(stereo_average_depth))

# #Cropping segmented areas from the images
# if available_box == True:
#     i=1 
#     for mask in masks:
#         #Cropping segment from original image
#         segment = Segment_Crop(mask.cpu().numpy(), plt.gca(), image)
#         #Finding area of segment in pixels
#         area_total = Pixel_Area(segment)
#         print("Mask {} Total Area (mm2): ".format(i) + str(area_total))
#         #Cropping depth map according to mask 
#         estimated_depth_segment = Segment_Crop(mask.cpu().numpy(), plt.gca, estimated_depth_map)
#         if stereo_available:
#             stereo_depth_segment = Segment_Crop(mask.cpu().numpy(), plt.gca, stereo_depth_map)
#         #Display image segment
#         plt.imshow(segment)
#         plt.title('mushroom')
#         plt.show()
#         #Display depth map segemnt
#         plt.imshow(estimated_depth_segment)
#         plt.title('estimated depth')
#         plt.show()
#         #Display depth map segemnt
#         if stereo_available:
#             plt.imshow(stereo_depth_segment)
#             plt.title('stereo depth')
#             plt.show()
#         i=i+1

# #Calculating the real area of the segmented object in the image
# Real_Area(image, estimated_average_depth, area_total)
# if stereo_available:
#     Real_Area(image, stereo_average_depth, area_total)