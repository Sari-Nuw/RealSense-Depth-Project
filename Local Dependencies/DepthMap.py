from PIL import Image
import numpy as np

#Takes a depth map array and converts it to a color depth map image
def DepthMap(Array, dist_min, dist_max):
    
    #Getting height and width of the image
    rows = len(Array)
    columns = len(Array[0])

    #Generating array to store rgb values
    pixels = np.zeros((rows,columns,3), 'uint8')

    for x in range(rows):
        for y in range (columns):
            #Getting depth of pixel
            depth = Array[x][y]

            #Removing out of bounds depths
            if depth > dist_max:
                depth = dist_max
            elif depth < dist_min:
                depth = dist_min
            
            #Calculating grayscale 'percentage'
            percentage = (depth - dist_min)/dist_max
            #Setting pixel values in grayscale
            pixels[x][y][0] = round((1-percentage)*255) 
            pixels[x][y][1] = round((1-percentage)*255)
            pixels[x][y][2] = round((1-percentage)*255)
    
    #Generating image from values
    pixels = np.ascontiguousarray(pixels)
    img = Image.fromarray(pixels, 'RGB')  

    #Return the image
    return img

#Return from depth map to frame values
def DepthMaptoFrame(image, min_dist, max_dist):

    #Initialize array of zeros the same size as the image as placeholders
    depth_array = np.zeros((image.shape[0],image.shape[1]))

    #Converting pixel color to depth based on a percentage of the max/min vlaues
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            percentage = 1-(image[i][j][0])/255
            depth = percentage*(max_dist-min_dist)
            depth_array[i][j] = depth + min_dist

    return depth_array