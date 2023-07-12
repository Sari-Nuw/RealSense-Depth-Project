import cv2
import numpy as np

def EstimatedDepthBackgroundRemoval(color_frame, estimated_depth_frame, min_dist, max_dist):
    
    rows = len(color_frame)
    columns = len(color_frame[0])

    cutoff_dist = (max_dist - min_dist)/2

    color_frame_copy = np.copy(color_frame)

    for x in range(rows):
        for y in range(columns):
            if estimated_depth_frame[x][y] >= cutoff_dist:
                color_frame_copy[x][y] = [0, 0, 0]

    cv2.imshow('Removed', color_frame_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return color_frame_copy