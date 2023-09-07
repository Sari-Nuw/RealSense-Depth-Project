import cv2

#Finding average depth of masked area
def Depth_Calculation(depth):
    #Initializing depth and number of points 
    total_depth = 0
    depth_points = 0
    #Iterating across the depth image pixel by pixel
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            #If depth is differnt than 0
            if depth[i][j] != 0:
                #Add depth to total depth and increment depth points
                total_depth = total_depth + depth[i][j]
                depth_points = depth_points + 1
    #Caluclating average depth
    average_depth = total_depth/depth_points
    return average_depth

#Finding object area in pixels
def Pixel_Area(segment):
    #Initialize area
    area_total = 0
    #Convert segment to grayscale
    gray_segment = cv2.cvtColor(segment,cv2.COLOR_RGBA2GRAY)
    #Threshold the image (tunr the image to black and  white)
    _, threshhold_segment = cv2.threshold(gray_segment.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
    #Find image contours
    contours, _ = cv2.findContours(threshhold_segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #For all contours find area and add to toal area (if more than 100 pixels in size)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 100:
            area_total = area_total + area
    return area_total

#Calculating object area. 
# Only works for a object directly in line of sight of the camera. Modifiable for different angles?
# From online: Distance to object (mm) = focal length (mm) * real object area (mm2) * image area (pixels) / object area (pixels) * sensor area (mm2)
# -> Real object area (mm2) = [Distance to object (mm) * object area (pixels) * sensor area (mm2)] / [focal length (mm) * image area (pixels)]
def Real_Area(image, average_depth, area_total):
    #From D345 camera specs focal length in mm, sensor area in mm^2
    focal_length = 1.93 
    sensor_width = 14.67
    sensor_height = 8.5
    sensor_area = sensor_width * sensor_height 
    #Size of image in pixels
    image_area = image.shape[0]*image.shape[1]

    #Calculating the object area
    object_area = (average_depth * area_total * sensor_area)/(focal_length *image_area)
    print('Object area (mm2): ' + str(object_area))

    return object_area