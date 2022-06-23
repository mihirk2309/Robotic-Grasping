#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

import numpy as np
from PIL import Image as PILImage
from cv_bridge import CvBridge, CvBridgeError

# IMAGE_TOPIC = "/camera/rgb/image_color"
# DEPTH_TOPIC = "/camera/depth/image_raw"

IMAGE_TOPIC = "/kinect2/sd/image_color_rect"
DEPTH_TOPIC = "/kinect2/sd/image_depth_rect"

def get_image(show=False):
    #print("CALLING GET_KINECT_IMAGE")
    rospy.init_node("kinect_subscriber")
    rgb = rospy.wait_for_message(IMAGE_TOPIC, Image)
    depth = rospy.wait_for_message(DEPTH_TOPIC, Image)

    # Convert sensor_msgs.Image readings into readable format
    bridge = CvBridge()
    rgb = bridge.imgmsg_to_cv2(rgb, rgb.encoding)
    depth = bridge.imgmsg_to_cv2(depth, depth.encoding)

    image = rgb
    image[:, :, 2] = depth
    if (show):
        im = PILImage.fromarray(image, 'RGB')
        
        #cv2.imwrite('savedimage.jpeg', img)    
        #print(im.format)
        #rospy.loginfo(im.format)


        im.show()

    return image


if __name__ == '__main__':
    while 1:
        # rate = rospy.Rate(1) # 1 Hz
        # # Do stuff, maybe in a while loop
        # rate.sleep() # Sleeps for 1/rate sec
    #for i in range(2):

        image = get_image(show=True)

