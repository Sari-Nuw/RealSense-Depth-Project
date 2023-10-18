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
import math

# Convert Yolo bb to Pascal_voc bb
def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [int(x1), int(y1), int(x2), int(y2)]

#CAN BE IMPROVED FOR SORTING SPEED, DOES NOT TAKE INTO CONSIDERATION MORE OR LESS BOXES BEING COUNTED
def box_sort(boxes,basis):
    temp = []
    #Iterate through the 'basis points'
    for point in basis:
        #Set minimum distance to infinity to find closest points
        min = float('inf')
        for center in boxes:
            #Calculating the nearest points
            distance = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            if distance < min:
                min = distance
                min_point = center
        #Setting temporary point 
        temp.append(min_point)
    return temp

#Path to picture folder
path = r'C:\Users\nuway\OneDrive\Desktop\Realsense Project\Workshop Trial\Picture Subset'

#To open and store the images
images = []
depth_maps = []
bounded_objects = []
i = 1 

#Opening the image and depth images (original image names changed from time to numebring to open)
for file in os.listdir(path):
    image = cv2.imread(r'C:\Users\nuway\OneDrive\Desktop\Realsense Project\Workshop Trial\Picture Subset\Picture ({}).png'.format(i))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    bounded_objects.append(image.copy())
    depth_map = cv2.imread(r'C:\Users\nuway\OneDrive\Desktop\Realsense Project\Workshop Trial\Estimated Depth Map Subset\Estimated Depth Map ({}).png'.format(i))
    depth_maps.append(depth_map)
    i += 1

print(len(images))
print(len(depth_maps))

#Pixel coordinates of objects
box_points = [[] for _ in range(len(images))]

#Saving the position of the object boxes
for i in range(len(images)):
    #Opening hand annotated yolo label files 
    file = open(r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Workshop Trial\Picture Subset Data\Picture ({}).txt".format(i+1))
    lines = file.readlines()
    for line in lines:
        #Saving the annotations and converting to pascal annotation (upper left and bottom right of rectangle)
        numbers = line.split()
        x1, y1, x2, y2 = float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4])
        points = yolo_to_pascal_voc(x1, y1, x2, y2,images[i].shape[1],images[i].shape[0])
        box_points[i].append(points)

#Sorting the bounding boxes for consistency 
for i in range(len(box_points)-1):   
    box_points[i+1] = box_sort(box_points[i+1],box_points[i])
    print('box points {}'.format(i))
    print(box_points[i])

#Drawing boundary boxes around the mushrooms
for i in range(len(images)):
        mush_num = 0
        for box in box_points[i]:           
            #To see bounding box around detected object
            cv2.rectangle(bounded_objects[i], (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(bounded_objects[i], 'box {}'.format(mush_num), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            mush_num += 1

#Loading the segment anything models from files
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

#Set to use GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Loading the segment anything models
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#Generating image masks based on box prompts 
image_masks = []
for i in range(len(images)):
    predictor = SamPredictor(sam)
    predictor.set_image(images[i])
    input_boxes = torch.tensor(box_points[i], device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    image_masks.append(masks)
    print(i)

#Finding average depth
for i in range(len(depth_maps)):
    #Calculate depth frame values from depth map colors and minimum/maximum distance (in mm). SET AS 0 AND 2000
    estimated_depth_frame = DepthMaptoFrame(depth_maps[i], 0, 2000)
    # if stereo_available:
    #     stereo_depth_frame = DepthMaptoFrame(stereo_depth_map, min_dist, max_dist)
    #Cropping the depth frame values according to the mask
    estimated_masked_depth = Segment_Crop(image_masks[i][0].cpu().numpy(), plt.gca(), estimated_depth_frame, depth = True)
    # if stereo_available:
    #     stereo_masked_depth = Segment_Crop(masks[0].cpu().numpy(), plt.gca(), stereo_depth_frame, depth = True)
    #Average depth calculated
    estimated_average_depth = Depth_Calculation(estimated_masked_depth)
    # if stereo_available:
    #     stereo_average_depth = Depth_Calculation(stereo_masked_depth)
    print("Estimated Average Depth {} (mm2): ".format(i) + str(estimated_average_depth))
    # if stereo_available:
    #     print("Stereo Average Depth (mm2): " + str(stereo_average_depth))


#Cropping segmented areas from the images
mask_sizes = [[] for _ in range(len(images))]
for i in range(len(images)):
    mask_num = 1 
    for mask in image_masks[i]:
        #Cropping segment from original image
        segment = Segment_Crop(mask.cpu().numpy(), plt.gca(), images[i])
        #Finding area of segment in pixels
        area_total = Pixel_Area(segment)
        mask_sizes[i].append(area_total)
        #print("Mask {} {} Total Area (mm2): ".format(i,mask_num) + str(area_total))
        #Cropping depth map according to mask 
        estimated_depth_segment = Segment_Crop(mask.cpu().numpy(), plt.gca, depth_maps[i])
        # if stereo_available:
        #     stereo_depth_segment = Segment_Crop(mask.cpu().numpy(), plt.gca, stereo_depth_map)
        #Display image segment
        # plt.imshow(segment)
        # plt.title('mushroom')
        # plt.show()
        #Display depth map segemnt
        # plt.imshow(estimated_depth_segment)
        # plt.title('estimated depth')
        # plt.show()
        #Display depth map segemnt
        # if stereo_available:
        #     plt.imshow(stereo_depth_segment)
        #     plt.title('stereo depth')
        #     plt.show()
        mask_num += 1

line1, line2, line3, line4 = [], [], [], []
for i in range(len(mask_sizes)):
    line1.append(mask_sizes[i][0])
    line2.append(mask_sizes[i][1])
    line3.append(mask_sizes[i][2])
    line4.append(mask_sizes[i][3])

y_axis = np.arange(1,len(mask_sizes)+1)

plt.plot(y_axis, line1, label = 'line 1')
plt.plot(y_axis, line2, label = 'line 2')
plt.plot(y_axis, line3, label = 'line 3')
plt.plot(y_axis, line4, label = 'line 4')

plt.legend()
plt.show()