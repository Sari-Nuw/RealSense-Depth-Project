from shapely.geometry import Polygon, mapping
from Depth_Estimation import *
from UliEngineering.Math.Coordinates import BoundingBox
from Environmental_Tracking import *
from Realtime_Utils import *

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

#To draw out the harvested clusters
visualise_harvest = False

# Set the paths for differnt folders
mushroom_architecture_selected = "mushroom_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
substrate_architecture_selected = "substrate_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
working_folder = "./results/" + mushroom_architecture_selected + "/"
configs_folder = "./configs/"
predicted_images = working_folder + 'predicted_images/'
#Path to images
test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Timelapse/Full1//"
if env_option:
    #Pathway to the environemntal files
    data_test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Timelapse/Experiment 3 Data//"
if annotation_option:
    #Name and pathway to the relevant annotation text file 
    annotations = get_annotations('hungary_annotations.txt')

# create the result folders
os.makedirs(working_folder,exist_ok=True)
os.makedirs(configs_folder,exist_ok=True)
os.makedirs(predicted_images,exist_ok=True)

#Establishing csv files
establish_cluster_sizing(working_folder)
establish_metrics(working_folder,'presort')
establish_metrics(working_folder,'postsort')

#Checking for available cuda/cpu
use_device = check_cuda()

#Loading models prediction and depth estimation models
mushroom_model,substrate_model,visualizer = load_models(configs_folder,mushroom_architecture_selected,substrate_architecture_selected,use_device)

#Saving images
images = []
stereo_depth_images = []

#Tracking image information
image_files = []
data = []

#Tracking clusters and cluster information 
polygons = []
polygons_info = []

#Establishing baseline for sorting
baseline = []

#Tracking brightness of images for filtering
list_of_brightness = []

#Saving pixel length of the substrate in images
detected_length_pixels = []
averaged_length_pixels = []

#Saving the annotation metrics
presort_metrics = []
postsort_metrics = []

#To track cluster sizes
cluster_segments = []

#To track cluster growth
lines = []

#From the farm substrate (50 cm)
substrate_real_size = 50

#Confidence thresholds
confidence_score_threshold = 0.5

#Image size in pixels
img_size = 0

for img_num in range(len(os.listdir(test_set_path))):
    if img_num > -1 and img_num < 1000:

        test_img = 'img ({}).JPG'.format(img_num+1)

        #check and adjust brightness
        list_of_brightness.append(brightness(test_set_path + test_img))
        if abs(list_of_brightness[-1] - np.mean(list_of_brightness))>4*np.std(list_of_brightness):
            print("Image with outlier brightness (lights on) detected and skipped: ", test_img)
            list_of_brightness.pop()
            continue

        # load the image
        img = mmcv.imread(test_set_path + test_img)
        substrate_img = img.copy()

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

        #Substrate segmentation inference
        substrate_result = inference_detector(substrate_model, img).pred_instances
                
        # calculate substrate length data
        detected_length_pixels.append(substrate_result[0]["bboxes"].cpu().numpy()[0][2] - substrate_result[0]["bboxes"].cpu().numpy()[0][0])

        # calculate the substrate length average
        averaged_length_pixels.append(sum(detected_length_pixels)/len(detected_length_pixels))

        # Mushroom segmentation inference
        image_result = inference_detector(mushroom_model, img)
        image_result = delete_low_confidence_predictions(image_result,confidence_score_threshold)
        image_result = delete_overlapping_with_lower_confidence(image_result,iou_threshold=0.7)
        image_result = delete_post_background_clusters(image_result,substrate_result)

        #Color correction of the images
        img = mmcv.image.bgr2rgb(img)
        substrate_img = img.copy()

        #Draw bounding boxes on substrate images
        for result in substrate_result:
            sub_result = result["bboxes"].cpu().numpy()[0]
            cv2.rectangle(substrate_img,(int(sub_result[0]),int(sub_result[1])),(int(sub_result[2]),int(sub_result[3])),(0,0,255),5)

        #Save substrate images
        os.makedirs(working_folder + "/Substrate/",exist_ok=True)
        cv2.imwrite(working_folder + "/Substrate/images ({}).JPG".format(img_num+1), cv2.cvtColor(substrate_img,cv2.COLOR_RGB2BGR))

        #saving image for processing and image file names 
        images.append(img)
        image_files.append(test_set_path + test_img)

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
                #Reducing the size of the coordinate list
                points = points[0::2]
                #Finding the concave hull (outline) of the mask 
                #Note: you can change the concativity of the hull to have a 'tighter' fit
                #A tighter fit can result in errors occuring due to the polygon 'overlapping' itself
                #This can be changed through the library (ctrl+click concave_hull_indexes)
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

                #Results info include: Cluster label, bbox height/width, absolute height/width, bbox coordinates, and cluster harvest marker [0] 
                results_info.append([cluster_label,pixel_cluster_height,pixel_cluster_width,absolute_cluster_height,absolute_cluster_width,cluster_bbox,[0]])

        #Saving the hull results for all the clusters in the image
        polygons.append(results)
        polygons_info.append(results_info)

        #Pre-sorting annotation metrics
        if annotation_option:
            mAP, mAR, F1 ,TP, FP, FN = get_annotation_metrics(annotations[img_num],polygons[-1])
            presort_metrics.append([mAP,mAR,F1,TP,FP,FN])
            write_metrics(working_folder,'presort',presort_metrics[-1],img_num)
            annotation_img = img.copy()
            for annotation in annotations[img_num]:
                centre = Polygon(annotation).centroid
                cv2.polylines(annotation_img, np.int32([annotation]), True, (255, 0, 0), 5)
                os.makedirs(working_folder + "/Annotated/",exist_ok=True)
                cv2.imwrite(working_folder + "/Annotated/image ({}).JPG".format(img_num+1), cv2.cvtColor(annotation_img,cv2.COLOR_RGB2BGR))

        #Sorting clusters for tracking
        if tracking_option:

            #Testing purposes
            i = 0
            unsorted_string = []
            unsorted_img = img.copy()
            for poly in polygons[-1]:
                unsorted_string.append(poly[0])
                centre = Polygon(poly).centroid
                cv2.polylines(unsorted_img, np.int32([poly]), True, (255, 0, 0), 5)
                cv2.putText(unsorted_img, 'Pred{} {}'.format(i,poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
                os.makedirs(working_folder + "/Unsorted/",exist_ok=True)
                cv2.imwrite(working_folder + "/Unsorted/image ({}).JPG".format(img_num+1), cv2.cvtColor(unsorted_img,cv2.COLOR_RGB2BGR))
                i += 1
            print('unsorted')
            print(unsorted_string)

            baseline_string = []
            for base in baseline:
                baseline_string.append(base[0][0])
            print('baseline')
            print(baseline_string)

            polygons,polygons_info,baseline,harvested = cluster_sort(polygons,polygons_info,baseline)

            sorted_string = []
            j=0
            for poly in polygons[-1]:
                sorted_string.append([poly[0],polygons_info[-1][j]])
                j += 1
            print('sorted')
            print(sorted_string)

        #Saving the images with outlines and estimated/stereo depth information
        #Copying the current image for processing
        img = np.copy(images[-1])
        image_copy = (img, cv2.COLOR_RGB2BGR)[0]
        full_image = np.copy(images[-1])
        if cluster_sizing_option:
            sizing_image = np.copy(images[-1])
        #Post sorting annotation metrics
        if annotation_option:
            mAP, mAR, F1, TP, FP, FN = get_annotation_metrics(annotations[img_num],polygons[-1])
            postsort_metrics.append([mAP,mAR,F1,TP,FP,FN])
            write_metrics(working_folder,'postsort',postsort_metrics[-1],img_num)
        #To track sizing lines
        numbering = 1
        j = 0
        #Polygons in each image
        for poly in polygons[-1]:
            #Draw lines
            if len(poly) > 1:
                poly.reshape(-1,1,2)
                #Getting the centre point of the polygons
                centre = Polygon(poly).centroid
                #Finding the bounding box of the polygon to save the image as its own unique section
                bounding = BoundingBox(poly)
                if cluster_sizing_option:
                    #Drawing the horizontal and vertical sizing lines on the image
                    segments,numbering = cluster_sizing(bounding,polygons_info[-1][j][2],polygons_info[-1][j][1],poly,sizing_image,numbering)
                    polygon_area = Polygon(poly).area
                    if dynamic_substrate_option:
                        absolute_cluster_area = pixel_absolute_area(polygon_area,averaged_length_pixels[-1],50)
                    else:
                        absolute_cluster_area = pixel_absolute_area(polygon_area,averaged_length_pixels[-1],50)
                    cluster_segments.append([img_num+1,j,polygon_area,absolute_cluster_area,polygons_info[-1][j],segments])
                    #Update dynamic cluster sizing csv
                    write_cluster_sizing(cluster_segments[-1],working_folder)
                #Isolate and save the cluster from the original image
                box_image,local_poly = process_cluster(image_copy,poly,bounding,working_folder,-1,j)
                #Saving the image with outlined clusters
                cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 5)
                cv2.putText(full_image, '{} {}'.format(j,poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
                if annotation_option:
                    do_nothing = 0
                    #annotation_iou(annotations[i],poly,full_image,centre)
                #Getting the estimated depth for estimated and stereo depths (NEEDS WORK)
                if stereo_option:
                    polygon_coordinates = tuple(tuple(map(int,tup)) for tup in mapping(Polygon(poly))['coordinates'][0])
                    #Estimating the avergae distance of the cluster from the camera
                    avg_stereo_depth = Stereo_Depth_Estimation(polygon_coordinates,stereo_depth_images[-1])
                    cv2.putText(full_image, str(avg_stereo_depth) + ' cm', (int(centre.x),int(centre.y)+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                #Saving the image information for an individual cluster in numpy array format
                if array_option:
                    save_cluster_array(sizing_image,poly,centre,box_image,local_poly,working_folder,img_num,j)
            j += 1
        #To draw clusters that are blocked       
        if visualise_harvest:
            if harvested != []:
                for poly in harvested:
                    poly.reshape(-1,1,2)
                    #Getting the centre point of the polygons
                    centre = Polygon(poly).centroid
                    #Saving the image with outlined clusters (harvested)
                    cv2.polylines(full_image, np.int32([poly]), True, (0, 0, 255), 5)
                    cv2.putText(full_image, '{}'.format(poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)        
            

        #Saving image in various forms
        save_image(working_folder,full_image,img_num)
        if cluster_sizing_option:
            #Saving sizing image
            save_sizing_image(working_folder,sizing_image,img_num)
            cluster_segments.append([])
            #Update dynamic cluster csv
            write_cluster_sizing(cluster_segments[-1],working_folder)
            #Saving the image information in numpy array format
            if array_option:
                save_image_array(full_image,polygons[-1],working_folder,img_num)

        print('Cluster Model Image {}'.format(img_num))

        #Equalizing polygon list
        #Finding longest set of polygons
        max_length = 0
        for poly in polygons:
            if len(poly) > max_length:
                max_length = len(poly)
        #Adding null points to make the list equal sizes
        i = 0
        for poly in polygons:
            while len(poly) < max_length:
                poly.append([0])
                polygons_info[i].append([0])
            i += 1

        #Gathering the information from individual clusters across images to be able to track their growth
        if tracking_option:
            lines = line_setup(polygons[-1],polygons_info[-1],lines,img_size)


#Extracting the environmental variables from the csv files and combining with image data
if env_option:
    environmental_variable_prep(data_test_set_path,working_folder,image_files, lines)
        
#Plotting the growth curves
if tracking_option:
    #Initializing x-axis
    x_axis = np.linspace(0,len(lines[-1]),num = len(lines[-1]))
    plot_growth(polygons,x_axis,lines)
