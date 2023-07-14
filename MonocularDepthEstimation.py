import cv2
import torch
#import matplotlib.pyplot as plt
import numpy as np

def MonocularDepthEstimation(color_frame,min_dist,max_dist):

    #Downloading the MiDas (Depth Estimation Algorithm)
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')

    #Connecting to GPU if available or CPU otherwise
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    midas.to(device)
    midas.eval()

    #Import transformation pipeline
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.dpt_transform
    
    #transform input for midas
    img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    #Midas depth prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = 'bicubic',
            align_corners = False
        ).squeeze()

    #Generating the depth map from the Midas prediction and normalizing the values
    depth_map_predicted = prediction.cpu().numpy()
    #Prediction returns inverted depth map so depth needs to be reversed
    depth_map_predicted = -depth_map_predicted
    #Depth map normalized from 0-2000 (same parameters of DepthMap function) but this is just a relative depth NOT ABSOLUTE
    depth_map_predicted = cv2.normalize(depth_map_predicted, None, int(min_dist), int(max_dist), norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    #np.savetxt('monocular.txt', depth_map_predicted)
    
    return depth_map_predicted