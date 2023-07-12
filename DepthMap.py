from PIL import Image
import numpy as np
import cv2

#Takes a depth map Array and converts it to a color depth map image
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
            
            #Generating rgb values
            percentage = (depth - dist_min)/dist_max
            #Color implementation of depth map
            # if percentage > 0.66:
            #     pixels[x][y][0] = round((1-percentage)/(1-0.66)*255) 
            #     pixels[x][y][1] = 0
            #     pixels[x][y][2] = 0
            # elif percentage > 0.33:
            #     pixels[x][y][0] = 255
            #     pixels[x][y][1] = round((0.66-percentage)/(0.66-0.33)*255)
            #     pixels[x][y][2] = 0
            # else:
            #     pixels[x][y][0] = 255 
            #     pixels[x][y][1] = 255
            #     pixels[x][y][2] = 255 - round((0.33-percentage/0.33)*255)

            #Greyscale Depthmap implementation
            pixels[x][y][0] = round((1-percentage)*255) 
            pixels[x][y][1] = round((1-percentage)*255)
            pixels[x][y][2] = round((1-percentage)*255)
    
    #Generating image from rgb values
    pixels = np.ascontiguousarray(pixels)
    img = Image.fromarray(pixels, 'RGB')  
    img.show()

    # Array = (Array/256).astype(np.uint8)
    # new_map = cv2.applyColorMap(Array, cv2.COLORMAP_JET)
    # cv2.imshow('heatmap', new_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #Return the image
    return img