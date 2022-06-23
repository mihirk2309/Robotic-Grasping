#!/usr/bin/env python
from distutils.log import debug
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import matplotlib.cm as cm
import cv2 
import numpy as np
from PIL import Image as PILImage

idx=0

IMAGE_TOPIC = "/kinect2/hd/image_color_rect"
DEPTH_TOPIC = "/kinect2/hd/image_depth_rect"


def kinect_rgbd_callback(rgb_data, depth_data):
    """
    Save raw RGB and depth input from Kinect V1
    :param rgb_data: RGB image
    :param depth_data: raw depth image
    :return: None
    """
    try:
        cv_rgb = cv_bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        cv_depth = cv_bridge.imgmsg_to_cv2(depth_data, "32FC1")

        cv_rgb_arr = np.array(cv_rgb, dtype=np.uint8)
        cv_depth_arr = np.array(cv_depth, dtype=np.float32)
        # cv_depth_arr = np.nan_to_num(cv_depth_arr)

        if(False):
            cv2.imshow("Depth", cv_depth)
            cv2.imshow("RGB", cv_rgb)
            cv2.waitKey(2000)

        plt.subplot(1, 2, 1)
        plt.imshow(cv_rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(cv_depth)
        plt.show()




        img = cv_rgb_arr.copy()
        depth_raw = cv_depth_arr.copy()

        gray = img.astype(np.uint8)
        depth_raw = depth_raw

        print(depth_raw.shape)
        print("Min , max :",np.min(depth_raw) , np.max(depth_raw) )
        
        for i in range(depth_raw.shape[0]):
            for j in range(depth_raw.shape[1]):
                if(depth_raw[i,j] > 2500):
                    depth_raw[i,j]  = 2500
        
        print("Min , max :",np.min(depth_raw) , np.max(depth_raw) )

        
        depth = (depth_raw)#.astype(np.uint8)
        depth = (depth_raw-500)/(2500-00)*255
        depth = (depth_raw).astype(np.uint8)




        print("Min , max :",np.min(depth) , np.max(depth) )
        rgd_image = gray
        rgd_image[:,:,2] = depth

        if(False):
            cv2.imshow("RGD", rgd_image)
            cv2.imshow("RGB", cv_rgb)
            cv2.waitKey(0)

        plt.subplot(2, 2, 1)
        plt.imshow(rgd_image)
        plt.subplot(2, 2, 2)
        plt.imshow(cv_depth)

        x, y, w, h = 500, 0, 800, 700  #(X,W) = (500,700) FOR SQUARE
        crop_rgd = rgd_image[y:y+h, x:x+w]
        crop_depth = cv_depth[y:y+h, x:x+w]

        plt.subplot(2, 2, 3)
        plt.imshow(crop_rgd)
        plt.subplot(2, 2, 4)
        plt.imshow(crop_depth)
        plt.show()
        #cv2.imwrite('RGD_image.png',crop_rgd)  #changes the color format

        # saves RG-D image 
        im_rgd = PILImage.fromarray(crop_rgd, 'RGB')
        im_rgd.save("img_rgd_pakad","PNG")

        # saves the captured RGB-IMAGE
        im_rgd = PILImage.fromarray(cv_rgb, 'RGB')
        im_rgd.save("cv_rgb","PNG")


    except CvBridgeError as e:
        print(e)



def get_image(show=False):
    global idx
    #print("CALLING GET_KINECT_IMAGE")
    rospy.init_node("kinect_subscriber")
    rgb = rospy.wait_for_message(IMAGE_TOPIC, Image)
    depth = rospy.wait_for_message(DEPTH_TOPIC, Image)

    # Convert sensor_msgs.Image readings into readable format
    print(type(rgb))
    print(type(depth))
 
    bridge = CvBridge()
    rgb_image = bridge.imgmsg_to_cv2(rgb, desired_encoding="rgb8")
    
    # depth.encoding = "mono16"
    depth_image = bridge.imgmsg_to_cv2(depth, desired_encoding='8UC1')
    depth_array = np.array(depth_image, dtype=np.float32)

    rgd_image = rgb_image
    rgd_image[:, :, 2] = depth_image

    print(type(rgb_image))
    print(type(depth_image))
      
    # view RGB and RGD image
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.subplot(1, 2, 2)
    plt.imshow(rgd_image)
    plt.show()

    rgb_image = rgb_image[380:1080 ,400:1600]   #y,y+h   x,x+w
    depth_image = depth_image[380:1080 ,400:1600]   #y,y+h   x,x+w

    if (show):
        cv2.imshow("RGB Image window", rgb_image)
        cv2.imshow("RGD Image window", rgd_image)
        cv2.waitKey(0)
    return 

    if (show):
        # im = PILImage.fromarray(image, 'RGB')
        # im.save("img_rgd%02i.png"%idx,"PNG")
        
        # im.save("/home/mihir/Btech_ws/src/grasp_ros/Images/img_rgd%02i.png"%idx,"PNG")


            #save RGB Image
        # im_rgb = PILImage.fromarray(rgb, 'RGB')
        # im_rgb.save("img_rgb%02i"%idx,"PNG")

            # Depth Image
        # imdepth = PILImage.fromarray(depth, 'L')
        # imdepth.save("img_depth%02i"%idx,"PNG")

            #To save image as png
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imwrite("depth_img%02i.png", depth_colormap)

        idx = idx + 1 
        #print(im.format)
        #rospy.loginfo(im.format)


        #imshow(im)

    return image



if __name__ == '__main__':
    
    rospy.init_node("kinect_subscriber")
    rgb = rospy.wait_for_message(IMAGE_TOPIC, Image)
    depth = rospy.wait_for_message(DEPTH_TOPIC, Image)
    cv_bridge = CvBridge()

    kinect_rgbd_callback(rgb,depth)
    
    #while 1:
        # rate = rospy.Rate(1) # 1 Hz
        # # Do stuff, maybe in a while loop
        # rate.sleep() # Sleeps for 1/rate sec
    #image = get_image(show=False)

 
