import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from MonocularDepthEstimation import *
from DepthMap import *
import time
from ultralytics import YOLO
from Segment_Crop import *
from Volume_Estimation import *

start = time.time()

sys.path.append("..")

#Reading image and converting to RGB from BGR
image = cv2.imread(r'Pictures\Picture Date 27-07-2023 Time 17_26_07.png')
depth_map = cv2.imread(r'Pictures\Stereo Depth Map Date 27-07-2023 Time 17_26_07.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model_path = r'runs\detect\train27\weights\last.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.1

results = model(image)[0]

box_points = []

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        points = [int(x1),int(y1),int(x2),int(y2)]
        box_points.append(points)
        #To see bounding box around detected object
        # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        # cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
print(box_points)

#To check if any objects were detected
if box_points == []:
    available_box = False
else:
    available_box = True

# #Show original image
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

#Generating image masks based on box prompts 
if available_box == True:
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_boxes = torch.tensor(box_points, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

#Finding avergae depth
if available_box == True:
    #Calculate depth frame values from depth map colors and minimum/maximum distance (in mm)
    depth_frame = DepthMaptoFrame(depth_map,0,2000)
    #Cropping the depth frame values according to the mask
    masked_depth = Segment_Crop(masks[0].cpu().numpy(), plt.gca(), depth_frame, depth = True)
    #Average depth calculated
    average_depth = Depth_Calculation(masked_depth)
    print(average_depth)

#Cropping segmented areas from the images
if available_box == True:
    for mask in masks:  
        #Cropping segment from original image
        segment = Segment_Crop(mask.cpu().numpy(), plt.gca(), image)
        #Finding area of segment in pixels
        area_total = Pixel_Area(segment)
        print(area_total)
        #Cropping depth map according to mask 
        depth_segment = Segment_Crop(mask.cpu().numpy(), plt.gca, depth_map)
        #Display image segment
        plt.imshow(segment)
        plt.title('mushroom')
        plt.show()
        #Display depth map segemnt
        plt.imshow(depth_segment)
        plt.title('depth')
        plt.show()

#Calculating the real area of the segmented object in the image
Real_Area(image, average_depth, area_total)

    