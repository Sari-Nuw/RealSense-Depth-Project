import pyrealsense2 as rs
import numpy as np

class DepthCamera:

    #Initilazing the intrinsic paramters of the camera object
    intrinsics_width = 0
    intrinsics_height = 0
    intrinsics_fx = 0
    intrinsics_fy = 0
    intrinsics_ppx = 0
    intrinsics_ppy = 0

    #Initializing pipeline of th camera object
    pipeline = 0

    #Initializing camera object
    def __init__(self):

        # # Configure depth and color streams
        # self.pipeline = rs.pipeline()
        # config = rs.config()

        # # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()

        # #Enabling streaming
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # # Start streaming
        # profile = self.pipeline.start(config)

        # #Getting intrinsics
        # intrinsic = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # #Setting intrinsics
        # self.intrinsics_width = intrinsic.width
        # self.intrinsics_height = intrinsic.height
        # self.intrinsics_fx = intrinsic.fx
        # self.intrinsics_fy = intrinsic.fy
        # self.intrinsics_ppx = intrinsic.ppx
        # self.intrinsics_ppy = intrinsic.ppy

        realsense_ctx = rs.context()
        connected_devices = []
        for i in range (len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            connected_devices.append(detected_camera)

        i = 0
        #self.pipeline = []
        pipelines = []
        config = []
        for device in connected_devices:
            # Configure depth and color streams
            #self.pipeline.append(rs.pipeline())
            pipelines.append(rs.pipeline())
            config.append(rs.config())
            config[i].enable_device(device)

            #Enabling streaming
            config[i].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config[i].enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            i+=1

        i = 0
        width = []
        height = []
        fx = []
        fy = []
        ppx = []
        ppy = []
        for device in config:
            # Start streaming
            #profile = self.pipeline[i].start(device)
            profile = pipelines[i].start(device)

            #Getting intrinsics
            intrinsic = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

            #Setting intrinsics
            width.append(intrinsic.width)
            height.append(intrinsic.height)
            fx.append(intrinsic.fx)
            fy.append(intrinsic.fy)
            ppx.append(intrinsic.ppx)
            ppy.append(intrinsic.ppy)

            i+=1

        #Setting intrinsics
        self.intrinsics_width = width
        self.intrinsics_height = height
        self.intrinsics_fx = fx
        self.intrinsics_fy = fy
        self.intrinsics_ppx = ppx
        self.intrinsics_ppy = ppy

        #Setting pipeline
        self.pipeline = pipelines
    

    #Getting frames for display
    def get_frame(self):
        #Get frames
        depth_images = []
        color_images = []
        for pipeline in self.pipeline:
            frames = pipeline.wait_for_frames()
            aligngned_frames = rs.align(rs.stream.color).process(frames)

            #Color and depth frame objects
            depth_frame = aligngned_frames.get_depth_frame()
            color_frame = aligngned_frames.get_color_frame()

            #Depth frame filtering to reduce noise/holes
            rs.decimation_filter().set_option(rs.option.filter_magnitude, 1)
            depth_frame = rs.decimation_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            depth_frame = rs.hole_filling_filter().process(depth_frame)

            #Converting image to array
            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())
            depth_images.append(np.asanyarray(depth_frame.get_data()))
            color_images.append(np.asanyarray(color_frame.get_data()))
            

        #If no color or depth images available
        if not depth_frame or not color_frame:
            return False, None, None
        #With color and depth images available
        return True, depth_images, color_images

    #Stop camera object
    def release(self):
        self.pipeline.stop()