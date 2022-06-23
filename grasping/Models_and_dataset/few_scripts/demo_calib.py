# code for calibration test

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import torch.nn as nn
from skimage import io
import numpy as np
import cv2


#main code

# Projection matrix of Kinect
P = np.array([[1081.372, 0.0, 959.5],
              [0.0, 1081.372, 539.5],
              [0.0, 0.0, 1.0]]
            )

# camera intrinsic matrix of Kinect v2
cameraMatrix = np.array([[1081.372, 0.0, 959.5],
                        [0.0, 1081.372, 539.5],
                        [0.0, 0.0, 1.0]]
                )



# main code
if __name__ == '__main__':
    # reading image using matplotlib

    image = plt.imread('/home/pranay/robotic-grasping-cornell-new/cv_rgb')
    image_crop = image[0:700, 300:1420]
    plt.imshow(image_crop)
    plt.show()

    # x_workbench_dist_mm, y_workbench_dist_mm = 1100, 705  
    # x_camera_pixel,y_camera_pixel = 1120, 700    # earlier few error
    x_workbench_dist_mm, y_workbench_dist_mm = 1100, 675
    x_camera_pixel,y_camera_pixel = 1120, 685
    
    mm_per_pixelX = x_workbench_dist_mm/x_camera_pixel 
    mm_per_pixelY = y_workbench_dist_mm/y_camera_pixel

    x_pixel_coord = int(input("Enter X coord:"))
    y_pixel_coord = int(input("Enter Y coood:"))
    print(x_pixel_coord*mm_per_pixelX)
    print(y_pixel_coord*mm_per_pixelY)





def Realcoordinates_from_pixelusingKmatrix():
    coordinates = []
    for i in range(3):
        a = int(input("Enter the coordinate position:"))
        print(a)
        coordinates.append(a)


    print("Entered values:", coordinates)
    print(np.linalg.inv(cameraMatrix).dot(np.array( [coordinates[0],coordinates[1],coordinates[2]] )))

    # reading image using matplotlib
    image = plt.imread('/home/pranay/robotic-grasping-cornell-new/cv_rgb')
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.show()

####################################################################################
    # Results

    # z = 1 is passed
    # p1 = (300,0)    # Real world =[-0.60987338 -0.49890325  1.        ]
    # p2 = (1420,0)     # Real world = [ 0.4258479  -0.49890325  1. 
    # p3 = (1420,700)   # Real world =  [0.4258479  0.14842256 1.        ]
    # p4 = (300,700)   # Real world =  [-0.60987338  0.14842256  1.        ]

    # x with = 1.035   
    # y height = 0.6473
    # ratio = 1.598

    # 1100 x 705 (mm x mm) real world measured (ratio = 1.56) 

####################################################################################