from shapely.geometry import Polygon, mapping
from UliEngineering.Math.Coordinates import BoundingBox
from Environmental_Tracking import *
from Mask_Generator_Utils_Depth import *

#Different utility options#
#Growth tracking option
tracking_option = False

#Using stereo depth data if available 
stereo_option = False

#Environmental option
env_option = False
if env_option:
    #Tracking needs to be on for environmental data
    tracking_option = True 

#Cluster sizing option
cluster_sizing_option = True
if cluster_sizing_option:
    tracking_option = True

#Save the prediction information in the form of numpy arrays
array_option = False
if array_option:
    cluster_sizing_option = True
    tracking_option = True

#Compare the predicted masks with the hand-annotated masks
annotation_option = False

#Measure the average substate size dynamically across the images
dynamic_substrate_option = True

# Set the paths for differnt folders
mushroom_architecture_selected = "mushroom_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
substrate_architecture_selected = "substrate_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
working_folder = "./results/" + mushroom_architecture_selected + "/"
configs_folder = "./configs/"
predicted_images = working_folder + 'predicted_images/'
#Path to images
test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Timelapse/Exp1//"
if env_option:
    #Pathway to the environemntal files
    data_test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Timelapse/Experiment 3 Data//"

# create the result folders
os.makedirs(working_folder,exist_ok=True)
os.makedirs(configs_folder,exist_ok=True)
os.makedirs(predicted_images,exist_ok=True)

#Checking for available cuda/cpu
use_device = check_cuda()

#Getting image files
#Images MUST be named 'img (1,2,3..).JPG'
test_set = get_test_set(test_set_path)
#To control test set size
test_set = test_set[90:110]
print(test_set)

#Loading models prediction and depth estimation models
mushroom_model,substrate_model,visualizer,pipe = load_models_depth(configs_folder,mushroom_architecture_selected,substrate_architecture_selected,use_device)

#Finding the average pixel size of the substrate in the images
averaged_length_pixels,detected_length_pixels = substrate_processing(substrate_model,test_set,test_set_path,working_folder)

#Iterating through the images and performing the predictions and depth estimations
images,image_files,data,polygons,polygons_info,polygons_depth_info,stereo_depth_images,estimated_depth_images,color_estimated_depth_images,img_size = image_processing_depth(0.5,test_set,test_set_path,predicted_images,averaged_length_pixels,mushroom_model,visualizer,pipe,stereo_option,env_option,dynamic_substrate_option)

#Sorting clusters for tracking
if tracking_option:
    polygons,polygons_info,polygons_depth_info = cluster_sort_depth(polygons,polygons_info,polygons_depth_info)

#Filter out clusters that don't appear frequently
polygons,polygons_info,polygons_depth_info = consistency_filter(polygons,polygons_info,polygons_depth_info)

if annotation_option:
    annotations = get_annotations('hungary_annotations.txt')

#Tracking which clusters are available from each image
cluster_track = [[] for _ in range(len(images))]
#To track cluster sizes
cluster_segments = []

#Saving the images with outlines and estimated/stereo depth information
#Polygons in each image
i = 0
for polygon in polygons:
    #Copying the current image for processing
    img = np.copy(images[i])
    depth_img_copy = np.copy(estimated_depth_images[i])
    image_copy = (img, cv2.COLOR_RGB2BGR)[0]
    full_image = np.copy(images[i])
    if cluster_sizing_option:
        sizing_image = np.copy(images[i])
    if annotation_option:
        for annotation in annotations[i]:
            cv2.polylines(full_image, np.int32([annotation]), True, (0, 255, 0), 10)
    #To track sizing lines
    numbering = 1
    j = 0
    for poly in polygon:
        #Draw lines
        if len(poly) > 1:
            poly.reshape(-1,1,2)
            #Getting the centre point of the polygons
            centre = Polygon(poly).centroid
            #Finding the bounding box of the polygon to save the image as its own unique section
            bounding = BoundingBox(poly)
            if cluster_sizing_option:
                #Drawing the horizontal and vertical sizing lines on the image
                segments,numbering = cluster_sizing(bounding,polygons_info[i][j][2],polygons_info[i][j][1],poly,sizing_image,numbering)
                if dynamic_substrate_option:
                    absolute_cluster_area = pixel_absolute_area(Polygon(poly).area,averaged_length_pixels[i],50)
                else:
                    absolute_cluster_area = pixel_absolute_area(Polygon(poly).area,averaged_length_pixels[-1],50)
                cluster_segments.append([i+1,j,absolute_cluster_area,polygons_info[i][j],segments,polygons_depth_info[i][j]])
            #Isolate and save the cluster from the original image
            box_image,depth_box_image,local_poly = process_cluster_depth(image_copy,depth_img_copy,poly,bounding,working_folder,i,j)
            #Saving the image with outlined clusters
            cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 5)
            cv2.putText(full_image, str(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
            if annotation_option:
                annotation_iou(annotations[i],poly,full_image,centre)
            if stereo_option:
                #Getting the estimated depth for estimated and stereo depths (NEEDS WORK)
                polygon_coordinates = tuple(tuple(map(int,tup)) for tup in mapping(Polygon(poly))['coordinates'][0])
                #avg_depth = Depth_Estimation(polygon_coordinates,estimated_depth_images[i])
                #Estimating the avergae distance of the cluster from the camera
                avg_stereo_depth = Stereo_Depth_Estimation(polygon_coordinates,stereo_depth_images[i])
                cv2.putText(full_image, str(avg_stereo_depth) + ' cm', (int(centre.x),int(centre.y)+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            #Saving the image information for an individual cluster in numpy array format
            if array_option:
                save_cluster_array_depth(sizing_image,poly,centre,box_image,depth_box_image,local_poly,working_folder,i,j)
            cluster_track[i].append(j)
        j += 1
    #Saving image in various forms
    save_image_depth(working_folder,full_image,sizing_image,estimated_depth_images,color_estimated_depth_images,cluster_sizing_option,i)
    if cluster_sizing_option:
        cluster_segments.append([])
        #Saving the image information in numpy array format
        if array_option:
            save_image_array_depth(full_image,depth_img_copy,polygon,working_folder,i)
    print(i)
    i += 1

#Gathering the information from individual clusters across images to be able to track their growth
if tracking_option:
    lines,x_axis = line_setup(polygons,img_size)

#Extracting the environmental variables from the csv files and combining with image data
if env_option:
    environmental_variable_prep(data_test_set_path,working_folder,image_files, lines)

#Writing information from cluster_segments to excel file
if cluster_sizing_option:
    write_cluster_sizing_depth(cluster_segments,working_folder)
        
#Plotting the growth curves
if tracking_option:
    plot_growth(polygons,x_axis,lines)
    
