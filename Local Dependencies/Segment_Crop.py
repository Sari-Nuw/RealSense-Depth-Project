import numpy as np
import cv2

def Segment_Crop(mask, ax, image, depth = False):
    #Manipulating the generated mask (shape and datatype changes)
    color = np.array([255, 255, 255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask_image.astype('uint8')
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY) 
    #Changing mask to a black and white image
    ret, mask_image = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)
    #Overlaying mask on image
    masked = cv2.bitwise_and(image,image,mask=mask_image)
    #Return the cropped depth
    if depth:
        return masked
    #Creating temporary image to "delete" background
    tmp = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    #Setting alpha (transparency) values to zero  
    _,alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    r, g, b = cv2.split(masked)
    rgba = [r,g,b, alpha]
    #Merging image color values with alpha values
    mask_image = cv2.merge(rgba,4)
    return mask_image