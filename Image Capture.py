import cv2
from realsense_depth import *
import math
import time
from datetime import datetime
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from DepthMap import * 
from Contours import *
from PointCloud import *
from PIL import Image
import pyrealsense2 as py2
from MonocularDepthEstimation import *
from EstimatedDepthBackgroundRemoval import *
import matplotlib.pyplot as plt
import os
import open3d as o3d
from sklearn.cluster import KMeans

#Initializing Camera and getting intrinsic parameters
camera = DepthCamera()
camera_intrinsics = [camera.intrinsics_width, camera.intrinsics_height, camera.intrinsics_fx, camera.intrinsics_fy, camera.intrinsics_ppx, camera.intrinsics_ppy]

#Skipping initial frames
time.sleep(3)

#Control how often an image is taken in minutes (interval = number of minutes)
interval = 1/6
wait = math.floor(interval*60)

#Expected minimum and maximum distances for the camera/image (in mm)
min_dist = 0
max_dist = 2000

#Setting up connection with Google Drive (sari.nuwayhid@ucdconnect.ie)
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

#Path under which images are saved
#path = r"c:\Users\nuway\OneDrive\Desktop\Realsense Project\Python Code 3.10\Images\\"
path = r'.\Images\\'
#Checking that path folder exists to save images locally. otherwise create the pathway.
if not os.path.exists(path):
    os.mkdir(path)

while True:
    try:
        #Start measuring time needed for image processing and upload
        start_time = time.time() 

        #Getting photo footage
        ret, depth_frame, color_frame = camera.get_frame()

        # img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)/255

        # img_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

        # #n_clusters determines number of colors with which to group image
        # kmeans = KMeans(n_clusters = 10, random_state=0).fit(img_2D)
        # clustered = kmeans.cluster_centers_[kmeans.labels_]

        # clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])

        # plt.imshow(clustered_3D)
        # plt.title('Original')
        # plt.show()

        # break

        # depth_frame = cv2.resize(depth_frame, (640,480), cv2.INTER_CUBIC)

        # #Making a copy of the camera depth frame and nomalizing it between 0-2000 for point cloud plotting
        # depth_frame_copy = np.copy(depth_frame)
        # depth_frame_copy = max_dist*((depth_frame_copy - np.min(depth_frame_copy))/(np.max(depth_frame_copy) - np.min(depth_frame_copy)))
        # depth_frame_copy =depth_frame_copy.astype('uint16')

        # print(np.min(depth_frame))
        # print(np.max(depth_frame))
        # plt.hist(depth_frame, bins=20)
        # plt.show()

        # print(np.min(depth_frame_copy))
        # print(np.max(depth_frame_copy))
        # plt.hist(depth_frame_copy, bins=20)
        # plt.show()

        # #Estimating the depth image from the color_frame image
        # moncular_depth_frame = MonocularDepthEstimation(color_frame, min_dist, max_dist)

        # print(np.min(moncular_depth_frame))
        # print(np.max(moncular_depth_frame))
        # plt.hist(moncular_depth_frame, bins=20)
        # plt.show()

        # EstimatedDepthBackgroundRemoval(color_frame, moncular_depth_frame, min_dist, max_dist)

        # #contour_img_foreground, contour_img = Contours(color_frame)

        # PointCloud(color_frame,moncular_depth_frame, camera_intrinsics)
        # PointCloud(color_frame,depth_frame_copy, camera_intrinsics)
        # break

        #Getting date and time
        now = datetime.now()
        now_str = now.strftime("Date %d-%m-%Y Time %H;%M;%S")

        #Resizing the actual depth frame due to "decimation" prost-processing and filtering techniqe in camera.get_frame()
        depth_frame = cv2.resize(depth_frame, (640,480), cv2.INTER_CUBIC)

        #Estimating the depth frame from the color_frame image
        monocular_depth_frame = MonocularDepthEstimation(color_frame, min_dist, max_dist)

        #Saving image
        written = cv2.imwrite(path+"Picture {}.png".format(now_str), color_frame)
        # Depth frame information 
        # cv2.imwrite(path+"Depth {}.png".format(now_str), depth_frame)
        # cv2.imwrite(path+"Estimated Depth {}.png".format(now_str), monocular_depth_frame)
        print(written)

        #Generating and saving the stereo depth map
        depth_img = DepthMap(depth_frame, min_dist, max_dist)
        depth_img.save(path+"Stereo Depth Map {}.png".format(now_str))
        
        #Generating and saving the estimated depth map
        estimated_depth_img = DepthMap(monocular_depth_frame, min_dist, max_dist)
        estimated_depth_img.save(path+"Estimated Depth Map {}.png".format(now_str))

        #Generating the contouors image for the foreground
        contour_img_foreground, contour_img = Contours(color_frame)
        #Saving the contoured image to the folder
        contour_img.save(path+"Contours {}.png".format(now_str))
        contour_img_foreground.save(path+"Contours Foreground {}.png".format(now_str))

        #Generating point cloud from color and estimated depth images
        estimated_pcd = PointCloud(color_frame,monocular_depth_frame, camera_intrinsics)
        o3d.io.write_point_cloud(path+"Estimated PCD {}.pcd".format(now_str), estimated_pcd)

        #Making a copy of the camera depth frame and nomalizing it between 0-2000 for point cloud plotting
        depth_frame_copy = np.copy(depth_frame)
        depth_frame_copy = (max_dist - min_dist)*((depth_frame_copy - np.min(depth_frame_copy))/(np.max(depth_frame_copy) - np.min(depth_frame_copy)))
        depth_frame_copy = depth_frame_copy.astype('uint16')
        #Generating point cloud from color and RealSense camera normalized depth images
        normalized_stereo_pcd = PointCloud(color_frame,depth_frame_copy, camera_intrinsics)
        o3d.io.write_point_cloud(path+"Normalized Stereo PCD {}.pcd".format(now_str), normalized_stereo_pcd)


        #Uploading files to Google Drive.
        upload_file_list = [path+"Picture {}.png".format(now_str),
                            path+"Stereo Depth Map {}.png".format(now_str), 
                            path+"Estimated Depth Map {}.png".format(now_str), 
                            path+"Contours {}.png".format(now_str), 
                            path+"Contours Foreground {}.png".format(now_str),
                            path+"Estimated PCD {}.pcd".format(now_str),
                            path+"Normalized Stereo PCD {}.pcd".format(now_str)]
        for upload_file in upload_file_list:
            gfile = drive.CreateFile({'parents':[{'id': '1vOjFlHQCKlqgM3Wrqe8P4Lo5TI5TN034'}]})
            #Read file and upload
            gfile.SetContentFile(upload_file)
            #Removing path from file names
            gfile['title'] = upload_file.replace(path,'')
            gfile.Upload()
        print("uploaded")

        #Complete measuring time needed for image processing and upload
        end_time = time.time()
        elapsed_time = end_time - start_time

        #Change wait time to account for image processing and upload
        wait = math.floor(wait - elapsed_time)
        #For processing time greater than wait time
        if wait < 0:
            wait = 0

        #Waiting to take another image
        time.sleep(wait)

    #Press Ctrl+C in terminal to stop the program
    except KeyboardInterrupt:
        print("exit")
        break