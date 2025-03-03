from shapely.geometry import Polygon
from Environmental_Tracking import *
from Information_Processing_Utilities import *
from Metrics_Utilities import *
from Model_Processing_Utilities import *
from Filtration_Utilities import *
from Sorting_utilities import *
from mmdet.apis import inference_detector
import mmcv
import copy
import sys

###Different utility options###

#Growth tracking option
tracking_option = False

#Environmental data processing
env_option = False
if env_option:
    #Tracking needs to be on for environmental data
    tracking_option = True 

#Generate images with cluster lengths/widths measured in pixels
cluster_sizing_option = True
if cluster_sizing_option:
    tracking_option = True

#Save the prediction information in the form of numpy arrays
array_option = True
if array_option:
    cluster_sizing_option = True
    tracking_option = True

#Compare the predicted masks with the hand-annotated masks
annotation_option = True

#Names of the instance segmentation models being used
mushroom_architecture_selected = "mushroom_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
#mushroom_architecture_selected = "mushroom_instance_segmentation_model_config"
substrate_architecture_selected = "substrate_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
#substrate_architecture_selected = "substrate_custom_config_mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco"
# Set the paths for differnt folders
working_folder = "./results/" + mushroom_architecture_selected + "/"
configs_folder = r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Python_Marigold\configs/"
# sys.path.append(configs_folder)
# configs_folder = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/configs/"
predicted_images = working_folder + 'predicted_images/'

#Path to images
test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Timelapse/Timelapse1//"
#test_set_path = r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Python_Marigold\Timelapse\Bucket/"

#Pathway to the environemntal files
if env_option:
    data_test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Timelapse/Experiment 3 Data//"

#Name and pathway to the relevant annotation text file 
if annotation_option:
    annotations, sorting_annotations = annotation_tracking("Full1.json")
    #annotations, sorting_annotations = annotation_tracking(r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Python_Marigold\Timelapse\Bucket\annotations.json")

# Creating the output folders
os.makedirs(working_folder,exist_ok=True)
os.makedirs(predicted_images,exist_ok=True)
os.makedirs(working_folder + "/Annotated/",exist_ok=True)
os.makedirs(working_folder + "/Unsorted/",exist_ok=True)
os.makedirs(working_folder + "/Substrate/",exist_ok=True)

#Establishing csv files
establish_cluster_sizing(working_folder)
establish_metrics(working_folder)
establish_mota(working_folder)

#Checking for available cuda/cpu
use_device = check_cuda()

#Loading prediction models
mushroom_model,substrate_model,visualizer = load_models(configs_folder,mushroom_architecture_selected,substrate_architecture_selected,use_device)

#Tracking images
images = []

#Tracking image information
image_files = []
data = []

#Tracking clusters and cluster information 
polygons = []
polygons_info = []
post_harvest_polygons_info_base = []

#Baseline for sorting
baseline = []

#Tracking brightness of images for filtering
list_of_brightness = []

#Saving pixel length of the substrate in images
detected_length_pixels = []
averaged_length_pixels = []

#Saving the annotation metrics
presort_metrics = []
postsort_metrics = []
TP_array_presort = []
FP_array_presort = []
FN_array_presort = []
TP_array_postsort = []
FP_array_postsort = []
FN_array_postsort = []

#Saving the annotation metrics without considering class label
presort_metrics_no_label = []
postsort_metrics_no_label = []
TP_array_presort_no_label = []
FP_array_presort_no_label = []
FN_array_presort_no_label = []
TP_array_postsort_no_label = []
FP_array_postsort_no_label = []
FN_array_postsort_no_label = []

# Sorting MOTA Metrics
mota = []
# False Positive, False Negative, ID Switch, Ground Truth
mota_metrics_50 = [[],[],[],[]]
mota_metrics_75 = [[],[],[],[]]
motaTracker_50 = []
motaTracker_75 = []

#To track cluster sizes
cluster_segments = []

#To track cluster growth
lines = []

#Confidence thresholds
confidence_score_threshold = 0.5
overlapping_iou_threshold = 0.2
post_harvest_occluded_iou_overlap = 0.5
harvest_margin = 0.5
harvest_threshold = 0.05

for img_num in range(len(os.listdir(test_set_path))):
    #To control which images are being processed
    if img_num > -1 and img_num < 300:

        test_img = 'img ({}).jpg'.format(img_num+1)

        #check and adjust brightness
        list_of_brightness.append(brightness(test_set_path + test_img))
        if abs(list_of_brightness[-1] - np.mean(list_of_brightness))>4*np.std(list_of_brightness):
            print("Image with outlier brightness (lights on) detected and skipped: ", test_img)
            list_of_brightness.pop()
            continue

        # load the image
        img = mmcv.imread(test_set_path + test_img)
        substrate_img = img.copy()

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

        #Color correction of the images
        img = mmcv.image.bgr2rgb(img)

        # # Chcecking substrate results and saving substrate image
        # process_substrate_results(img,substrate_result,working_folder,img_num)

        #saving image for processing and image file names 
        images.append(img)
        image_files.append(test_set_path + test_img)

        # show the results before filtering
        visualizer.add_datasample(
            'result',
            img,
            data_sample=image_result,
            draw_gt = None,
            wait_time=0,
            out_file=predicted_images + "before_filtration_prediction_" + test_img,
            pred_score_thr=confidence_score_threshold
        )

        # Result filters
        image_result = delete_low_confidence_predictions(image_result,confidence_score_threshold)
        image_result = delete_overlapping_with_lower_confidence(image_result,overlapping_iou_threshold)
        if post_harvest_polygons_info_base:
            image_result = delete_post_background_clusters(image_result,substrate_result,post_harvest_polygons_info_base,post_harvest_occluded_iou_overlap)

        #Processing of reuslts for use in different data structures
        results, results_info = process_results(image_result,averaged_length_pixels,substrate_real_size = 50)     

        #Saving the hull results for all the clusters in the image
        polygons.append(results)
        polygons_info.append(results_info)

        #Pre-sorting annotation metrics
        if annotation_option:
            # Processing instance segmentation metrics after filtration/before sorting
            mAP, mAR, AP50, AP75, F1 ,TP_array_presort, FP_array_presort, FN_array_presort = get_annotation_metrics(annotations[img_num],polygons[-1],polygons_info[-1],TP_array_presort, FP_array_presort, FN_array_presort)
            mAP_no_label, mAR_no_label, AP50_no_label, AP75_no_label, F1_no_label, TP_array_presort_no_label, FP_array_presort_no_label, FN_array_presort_no_label = get_annotation_metrics_no_label(annotations[img_num],polygons[-1],polygons_info[-1],TP_array_presort_no_label, FP_array_presort_no_label, FN_array_presort_no_label)
            presort_metrics.append([mAP,mAR, AP50, AP75, F1,TP_array_presort[-1],FP_array_presort[-1],FN_array_presort[-1]])
            presort_metrics_no_label.append([mAP_no_label, mAR_no_label, AP50_no_label, AP75_no_label, F1_no_label, TP_array_presort_no_label[-1], FP_array_presort_no_label[-1], FN_array_presort_no_label[-1]])
            write_metrics(working_folder,'Presort',presort_metrics[-1],img_num)
            write_metrics(working_folder,'Presort No Label',presort_metrics_no_label[-1],img_num)
            # Saving image with outlined annotations
            save_annotation_image(img,working_folder,sorting_annotations,img_num)

        #Sorting clusters for tracking
        if tracking_option:

            # Filter out recognized harvested clusters
            polygons[-1], polygons_info[-1], to_delete = harvest_filter(polygons[-1],polygons_info[-1],baseline,harvest_margin,harvest_threshold)

            image_result = image_result.cpu().numpy().to_dict()
            ## delete from all components of the result variable that are post-harvesting area
            image_result["pred_instances"]["bboxes"] = np.delete(image_result["pred_instances"]["bboxes"],to_delete, axis=0)
            image_result["pred_instances"]["scores"] = np.delete(image_result["pred_instances"]["scores"],to_delete, axis=0)
            image_result["pred_instances"]["masks"] = np.delete(image_result["pred_instances"]["masks"],to_delete, axis=0)
            image_result["pred_instances"]["labels"] = np.delete(image_result["pred_instances"]["labels"],to_delete, axis=0)
            image_result = dict_to_det_data_sample(image_result)

            # Show results after filtering and before sorting
            save_unsorted_image(img,polygons,working_folder,img_num)

            #Sorting the clusters
            polygons,polygons_info,baseline,to_delete = cluster_sort(polygons,polygons_info,baseline)

            image_result = image_result.cpu().numpy().to_dict()
            ## delete from all components of the result variable that are 'new' mature clusters
            image_result["pred_instances"]["bboxes"] = np.delete(image_result["pred_instances"]["bboxes"],to_delete, axis=0)
            image_result["pred_instances"]["scores"] = np.delete(image_result["pred_instances"]["scores"],to_delete, axis=0)
            image_result["pred_instances"]["masks"] = np.delete(image_result["pred_instances"]["masks"],to_delete, axis=0)
            image_result["pred_instances"]["labels"] = np.delete(image_result["pred_instances"]["labels"],to_delete, axis=0)
            image_result = dict_to_det_data_sample(image_result)

            # show the results after filtering
            visualizer.add_datasample(
                'result',
                img,
                data_sample=image_result,
                draw_gt = None,
                wait_time=0,
                out_file=predicted_images + "after_filtration_prediction_" + test_img,
                pred_score_thr=confidence_score_threshold
            )

        #Copying the current image for processing
        image_copy = (img, cv2.COLOR_RGB2BGR)[0]
        full_image = np.copy(img)
        if cluster_sizing_option:
            sizing_image = np.copy(img)

        #Post sorting annotation metrics
        if annotation_option:
            mAP, mAR, AP50, AP75, F1, TP_array_postsort, FP_array_postsort, FN_array_postsort = get_annotation_metrics(annotations[img_num],polygons[-1],polygons_info[-1],TP_array_postsort, FP_array_postsort, FN_array_postsort)
            mAP_no_label, mAR_no_label, AP50_no_label, AP75_no_label, F1_no_label, TP_array_postsort_no_label, FP_array_postsort_no_label, FN_array_postsort_no_label = get_annotation_metrics_no_label(annotations[img_num],polygons[-1],polygons_info[-1],TP_array_postsort_no_label, FP_array_postsort_no_label, FN_array_postsort_no_label)
            mota_metrics_50, mota_metrics_75, motaTracker_50, motaTracker_75 = get_sorting_metrics(sorting_annotations[img_num],polygons[-1],mota_metrics_50,mota_metrics_75,motaTracker_50,motaTracker_75)
            write_mota(working_folder,mota_metrics_50, mota_metrics_75,img_num)
            postsort_metrics.append([mAP,mAR, AP50, AP75, F1,TP_array_postsort[-1],FP_array_postsort[-1],FN_array_postsort[-1]])
            postsort_metrics_no_label.append([mAP_no_label, mAR_no_label, AP50_no_label, AP75_no_label, F1_no_label, TP_array_postsort_no_label[-1], FP_array_postsort_no_label[-1], FN_array_postsort_no_label[-1]])
            write_metrics(working_folder,'Postsort',postsort_metrics[-1],img_num)
            write_metrics(working_folder,'Postsort No Label',postsort_metrics_no_label[-1],img_num)
        
        #Saving the images with predicted polygons
        #Polygons from current image
        j = 0
        for poly in polygons[-1]:
            #Draw lines
            if len(poly) > 1:
                poly.reshape(-1,1,2)
                #Getting the centre point of the polygons
                centre = Polygon(poly).centroid
                if cluster_sizing_option:
                    #To track sizing lines
                    numbering = 1
                    #Drawing the horizontal and vertical sizing lines on the image
                    segments,numbering = cluster_sizing(polygons_info[-1][j][5],polygons_info[-1][j][2],polygons_info[-1][j][1],poly,sizing_image,numbering)
                    #Calculating estimated absolute cluster area from pixel area
                    absolute_cluster_area = pixel_absolute_area(Polygon(poly).area,averaged_length_pixels[-1],substrate_real_size = 50)
                    #Updating cluster segments
                    cluster_segments.append([img_num+1,j,Polygon(poly).area,absolute_cluster_area,polygons_info[-1][j],segments])
                    #Update dynamic cluster sizing csv
                    write_cluster_sizing(cluster_segments[-1],working_folder)

                #Saving the image with outlined clusters
                cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 10)
                #cv2.putText(full_image, '{} {}'.format(j,poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)
                cv2.putText(full_image, '{}'.format(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)

                #Saving the image information for an individual cluster in numpy array format
                if array_option:
                    #Isolate and save the cluster from the original image
                    box_image,local_poly = process_cluster(image_copy,poly,polygons_info[-1][j][5],working_folder,img_num,j)
                    save_cluster_array(sizing_image,poly,centre,box_image,local_poly,working_folder,img_num,j)
            j += 1    

        #Saving image in various forms
        save_image(working_folder,full_image,img_num)
        if cluster_sizing_option:
            #Saving sizing image
            save_sizing_image(working_folder,sizing_image,img_num)
            #Searate each image in cluster csv
            write_cluster_sizing([],working_folder)
            #Saving the image information in numpy array format
            if array_option:
                save_image_array(full_image,polygons[-1],working_folder,img_num)

        #Equalizing polygon list
        polygons, polygons_info = equalize_polygons(polygons,polygons_info)

        #Creating post-processing bbox baseline
        if not post_harvest_polygons_info_base:
            post_harvest_polygons_info_base = copy.deepcopy(polygons_info[-1])
        else:
            for i in range(len(polygons_info[-1])):
                if polygons_info[-1][i]==[0]:
                    continue
                if i<len(post_harvest_polygons_info_base):
                    post_harvest_polygons_info_base[i] = copy.deepcopy(polygons_info[-1][i])
                else:
                    post_harvest_polygons_info_base.append(copy.deepcopy(polygons_info[-1][i]))

        #Gathering the information from individual clusters across images to be able to track their growth
        if tracking_option:
            lines = line_setup(polygons[-1],lines,img.size/3)

        print('Image {}'.format(img_num+1))

#Extracting the environmental variables from the csv files and combining with image data
if env_option:
    environmental_variable_prep(data_test_set_path,working_folder,image_files, lines)


        
#Plotting the growth curves
if tracking_option:
    plot_growth(polygons,lines,working_folder)
