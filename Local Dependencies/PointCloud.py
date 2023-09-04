import open3d as o3d
import cv2

def PointCloud(color_frame, depth_frame, camera_intrinsics):

    #Changing color frame from BGR to RGB fro processing
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

    #Generating open3d images from depth and color frames
    depth_img = o3d.geometry.Image(depth_frame)
    color_img = o3d.geometry.Image(color_frame)

    #Generating the color/depth images
    rgbd_image= o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, convert_rgb_to_intensity = False)
    #Setting up the open3d camera parameter according to the camera
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(camera_intrinsics[0], camera_intrinsics[1], camera_intrinsics[2], camera_intrinsics[3], camera_intrinsics[4], camera_intrinsics[5])
    #Generating point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    #Flipping orientations
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    #Visualizing point cloud
    o3d.visualization.draw_geometries([pcd], width = camera_intrinsics[0], height = camera_intrinsics[1])

    return pcd