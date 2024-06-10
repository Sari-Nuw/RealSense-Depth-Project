import cv2
from realsense_depth import *

#Initializing Camera and getting intrinsic parameters
camera = DepthCamera()

#To calculate average depth for smoothness
depth_array = []

while True:
    try:
        #Getting photo footage
        ret, depth_frame, color_frame = camera.get_frame()

        #Resizing the actual depth frame due to "decimation" post-processing and filtering techniqe in camera.get_frame()
        depth_frame = cv2.resize(depth_frame, (640,480), cv2.INTER_CUBIC)

        depth_frame = np.clip(depth_frame,0,8000)
        depth_frame = depth_frame*1.8

        depth_array.append(depth_frame)

        #Preserving last 10 frames
        if len(depth_array) > 10:
            depth_array.pop(0)

        #Getting average depth across previoud 10 frames
        avg_depth = sum(depth_array)/10

        #avg_depth = np.interp(avg_depth, (avg_depth.min(), avg_depth.max()), (0, 8000))

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(avg_depth, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow("Depth Video", depth_colormap)
        cv2.imshow("Color Video", color_frame)
        cv2.waitKey(1)

    #Press Ctrl+C in terminal to stop the program
    except KeyboardInterrupt:
        print("exit")
        break