import cv2
import numpy as np
from MonocularDepthEstimation import *
from MonocularDepthEstimationMarigold import *
import matplotlib.pyplot as plt

def Depth_Estimation(polygon_coordinates,monocular_depth_frame):

    num_points = len(polygon_coordinates)
    # print(monocular_depth_frame)
    # print(polygon_coordinates)
    poly_sum = 0 
    for coordinate in polygon_coordinates:
        poly_sum = poly_sum + monocular_depth_frame[coordinate[1],coordinate[0]]
    #calculating average depth and converting to cm
    avg_depth = round(poly_sum/(num_points*10),1)
    print(avg_depth)

    return avg_depth

def Stereo_Depth_Estimation(polygon_coordinates,img):

    num_points = len(polygon_coordinates)
    poly_sum = 0 
    for coordinate in polygon_coordinates:
        poly_sum = poly_sum + img[coordinate[1],coordinate[0]]
    #calculating average depth and converting to cm
    avg_depth = round(poly_sum/(num_points*10),1)
    print(avg_depth)

    return avg_depth
