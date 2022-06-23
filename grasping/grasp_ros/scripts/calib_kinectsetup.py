#! /usr/bin/env python

# My ROS node 
from __future__ import absolute_import
from calendar import c
from distutils.log import debug


import os        
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import torch.nn as nn
from skimage import io
import numpy as np
import cv2
import cv2.aruco as aruco
from shapely.geometry import Polygon
from PIL import Image as PILImage


from lib.graspingnetwork.grasp_dataset import GraspDataset
from lib.graspingnetwork.network import GraspNet
from lib.graspingnetwork.available_gpus import get_available_gpus
from lib.prediction import demo_working

#ROS imports
import rospy
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import geometry_msgs.msg
from grasp_ros.msg import grasp
import serial
import time


#Model_path = '/home/pranay/catkin_ws/src/grasp_ros/src_model/grasping_coep/models'
IMAGE_TOPIC = "/kinect2/hd/image_color_rect"
DEPTH_TOPIC = "/kinect2/hd/image_depth_rect"



def rectangle_onVideo(x1,y1,x2,y2):

    cap = cv2.VideoCapture("/home/pranay/catkin_ws/src/grasp_ros/scripts/calib_kinectsetup/name")  #path

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 4)
            cv2.imshow('Frame',frame)     # Display the resulting frame

            if cv2.waitKey(25) & 0xFF == ord('q'):      # Press Q on keyboard to  exit
                break
        else: 
            break

    cap.release()  # When everything done, release the video capture object
    cv2.destroyAllWindows() # Closes all the frames





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

        #cv2.imshow("RGB", cv_rgb)
        x1,y1 = 500,150
        x2,y2 = 1050,700
        calib_frame = cv2.rectangle(cv_rgb, (x1,y1), (x2,y2), (0,0,0), 4) # draw rectange
        cv2.imshow('Calib_Frame',calib_frame)     # Display the resulting frame
        cv2.waitKey(10)
        

    except CvBridgeError as e:
        print(e)




if __name__ == '__main__':
    
    rospy.init_node("calib_kinectsetup", anonymous=True)

    cv_bridge = CvBridge()
    #rate = rospy.Rate(20.0) # MUST be more then 2Hz
    
    #print("Calibration Test:")

    while(1):
        
        rgb = rospy.wait_for_message(IMAGE_TOPIC, Image)
        depth = rospy.wait_for_message(DEPTH_TOPIC, Image)
        kinect_rgbd_callback(rgb,depth)

        if cv2.waitKey(25) & 0xFF == ord('q'):      # Press Q on keyboard to  exit
            break

        #rate.sleep()
        
       






# if __name__ =="__main__":
#     # initialize ros node
#     rospy.init_node("grasping_node")

#     # Bridge to convert ROS Image type to OpenCV Image type
#     cv_bridge = CvBridge()
#     cv2.WITH_QT = False
#     # Get camera calibration parameters
#     cam_param = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo, timeout=None)

#     # Subscribe to rgb and depth channel
#     image_sub = message_filters.Subscriber("/camera/rgb/image_rect_color", Image)
#     depth_sub = message_filters.Subscriber("/camera/depth_registered/image", Image)
#     ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.1)
#     ts.registerCallback(kinect_rgbd_callback)
#     rospy.spin()

#     demo_working.my_single_prediction(3)  # for realworld prediction.
    
#     test_working = 0
#     if(test_working ==1):
#         for id in range(177):
#             demo_working.my_single_prediction(2,id)   #show prediction on all images 
