#! /usr/bin/env python
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from grasp_ros.msg import grasp
import serial
import time

pose = grasp()

local_pos_pub = rospy.Publisher('arm/grasp_pose', grasp, queue_size=1)

def position_control():
    rospy.init_node('pub_node', anonymous=True)
    rate = rospy.Rate(20.0) # MUST be more then 2Hz
    local_pos_pub.publish(pose)
    rate.sleep()
   

if __name__=='__main__':
    while 1:
        position_control()
