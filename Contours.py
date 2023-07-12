import cv2
from PIL import Image
import numpy as np
from rembg import remove

#Lighting Threshholds (integer from 0 - 255)
threshold1 = 100
threshold2 = 200

def Contours(image):

    #Converting from a color image to greyscale for canny algorithm
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Canny edge detection to generate edges from the greyscale image 
    edges = cv2.Canny(img_gray,threshold1,threshold2)
    #Show edge image
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #Generating image from the image array
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    base = Image.fromarray(image_rgb)
    #Generating edge image and converting to "RGBA" mode to be compatible for blending
    overlay = Image.fromarray(edges)
    overlay = overlay.convert('RGBA')
    #Loading the pixel values of the overlay image
    pixels = overlay.load()
    #Change the alpha/transparency over all the black pixels to 0 (white pixels alpha remains 255)
    for x in range(overlay.size[0]):
       for y in range(overlay.size[1]):
           if pixels[x,y][0] <= 126:
               pixels[x,y] = (0, 0, 0, 0)

    #Overlaying the edges on top of the original image
    base.paste(overlay, (0,0), overlay)
    base.show()

    #Removing background from the contoured image
    img_foreground = remove(base)
    img_foreground.show()

    return img_foreground, base