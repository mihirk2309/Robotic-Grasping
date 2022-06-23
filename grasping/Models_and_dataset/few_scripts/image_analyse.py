import cv2
from matplotlib import pyplot as plt
import numpy as np

# img  = cv2.imread("few_scripts/Images/Images/img_rgb00.png")
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow("test",img)
# cv2.imshow("test2",imgGray)
# cv2.waitKey(0)




# reads an input image
img = cv2.imread("few_scripts/Images/Images/img_depth03.png",0)
print(type(img))

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])

# show the plotting graph of an image
plt.plot(histr)
plt.show()


#cv2.waitKey(0)