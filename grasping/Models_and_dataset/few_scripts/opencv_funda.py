import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# img = cv2.imread('/home/pranay/robotic-grasping-cornell-new/cv_rgb')
# cv2.imshow('image', img)
# cv2.waitKey(2000)

# reading image using matplotlib
image = plt.imread('/home/pranay/robotic-grasping-cornell-new/cv_rgb')
plt.subplot(1, 1, 1)
plt.imshow(image)
plt.show()

#print image specs
print(image.shape)
height, width, channel = image.shape[0:3]
print('Image height: ', height)
print('Image width: ', width)
print('Image channel: ', channel)


# canny edge detection 
img = cv2.imread('/home/pranay/robotic-grasping-cornell-new/cv_rgb')
edges = cv2.Canny(img,490,500)
 
cv2.imshow("Edge Detected Image", edges)
 
cv2.waitKey(0) # waits until a key is pressed