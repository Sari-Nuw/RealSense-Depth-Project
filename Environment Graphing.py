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

###Different utility options###

#Growth tracking option
tracking_option = True

#Environmental data processing
env_option = False
if env_option:
    #Tracking needs to be on for environmental data
    tracking_option = True 

#Compare the predicted masks with the hand-annotated masks
annotation_option = True

cluster_sizing_option = True

# Set the paths for differnt folders
working_folder = "./results/Env Trials/"
#working_folder = "./results/Lab Mushrooms/"

#Path to images
test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Env Trials/May 19 - May 26 2025/Top View/Top View Test/Images/"
#test_set_path = r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\Python_Marigold\Timelapse\Lab Mushrooms/"

#Pathway to the environemntal files
if env_option:
    data_test_set_path = r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Env Trials/Feb 26 - March 08 2024/"

#Name and pathway to the relevant annotation text file
if annotation_option:
    #annotations, sorting_annotations = annotation_tracking(r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Env Trials/June 30 - July 05 Side/annotations.json")
    annotations, sorting_annotations = annotation_tracking(r"C:/Users/nuway/OneDrive/Desktop/Realsense Project/Python_Marigold/Env Trials/May 19 - May 26 2025/Top View/Top View Test/annotations.json")

# Creating the output folders
os.makedirs(working_folder,exist_ok=True)

#Checking for available cuda/cpu
use_device = check_cuda()

#Tracking images
images = []

#Tracking image information
image_files = []
data = []
polygons = []
polygons_info = []
baseline = []

#To track cluster growth
lines = []

#Cluster size in pixels
cluster_segments = []
establish_cluster_sizing(working_folder)

for img_num in range(len(os.listdir(test_set_path))):
    #To control which images are being processed
    if img_num > -1 and img_num < 50:

        test_img = 'img ({}).png'.format(img_num+1)

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

        #Color correction of the images
        img = mmcv.image.bgr2rgb(img)

        #saving image for processing and image file names 
        images.append(img)
        image_files.append(test_set_path + test_img)  

        #Copying the current image for processing
        image_copy = (img, cv2.COLOR_RGB2BGR)[0]
        full_image = np.copy(img)
        if cluster_sizing_option:
            sizing_image = np.copy(img)

        annotation_list = []
        for annotation in annotations[img_num]:
            annotation_list.append(annotation[0])
        polygons.append(annotation_list)
        polygons_info.append(annotation_list)

        #Sorting the clusters
        polygons,polygons_info,baseline = cluster_sort_lab(polygons,polygons_info,baseline)
        
        #Saving the images with predicted polygons
        #Polygons from current image
        j = 0
        for poly in polygons[img_num]:
            #Draw lines
            #poly = poly[0]
            if len(poly) > 1:
                #Getting the centre point of the polygons
                centre = Polygon(poly).centroid
                if cluster_sizing_option:
                    #To track sizing lines
                    numbering = 1
                    #Drawing the horizontal and vertical sizing lines on the image
                    bbox = [Polygon(poly).bounds[0],Polygon(poly).bounds[1],Polygon(poly).bounds[2],Polygon(poly).bounds[3]]
                    x_diff = Polygon(poly).bounds[2] - Polygon(poly).bounds[0]
                    y_diff = Polygon(poly).bounds[3] - Polygon(poly).bounds[1]
                    segments,numbering = cluster_sizing(bbox,x_diff,y_diff,poly,sizing_image,numbering)
                    #Updating cluster segments
                    cluster_segments.append([img_num+1,j,Polygon(poly).area,'cluster area',['-1','-1','-1','-1','-1','-1'],segments])
                    #Update dynamic cluster sizing csv
                    write_cluster_sizing(cluster_segments[-1],working_folder)
                #Saving the image with outlined clusters
                cv2.polylines(full_image, np.int32([poly]), True, (255, 0, 0), 10)
                cv2.putText(full_image, '{}'.format(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)

            j += 1    

        #Saving image in various forms
        save_image(working_folder,full_image,img_num)
        if cluster_sizing_option:
            #Saving sizing image
            save_sizing_image(working_folder,sizing_image,img_num)
            #Searate each image in cluster csv
            write_cluster_sizing([],working_folder)

        #Equalizing polygon list
        polygons, polygons_info = equalize_polygons(polygons,polygons_info)

        #Gathering the information from individual clusters across images to be able to track their growth
        if tracking_option:
            lines = line_setup(polygons[img_num],lines,img.size/3)

        print('Image {}'.format(img_num+1))

#Extracting the environmental variables from the csv files and combining with image data
if env_option:
    #environmental_variable_prep(data_test_set_path,working_folder,image_files, lines)
    environmental_variable_prep_temp_hum(data_test_set_path,working_folder,image_files, lines)

#Writing csv file with growth curves
write_growth(lines,working_folder)
        
#Plotting the growth curves
if tracking_option:
    plot_growth(polygons,lines,working_folder)
