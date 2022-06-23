#! /usr/bin/env python
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from grasp_ros.msg import grasp
import serial
import time
import numpy as np

pose = grasp()
new_pose = grasp()
prev_pose = grasp()
zero_pose = grasp()



def pose_cb(Pose):
    global new_pose 
    new_pose = Pose
    #print(new_pose)

def next_pose_avail():
    if (prev_pose.graspx!=new_pose.graspx or prev_pose.graspy!=new_pose.graspy or prev_pose.theta!=new_pose.theta )and (new_pose.graspx!=0 or new_pose.graspy!=0 or new_pose.theta!=0):
        return True
    else:
        return False

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

#subscribe to the Coordinate publisher, which gives the grasp pose from model
rospy.Subscriber('arm/grasp_pose', grasp, pose_cb)
local_pos_pub = rospy.Publisher('arm/grasp_pose', grasp, queue_size=1)

print("")
print("----------------------------------------------------------")
print("Welcome to the MoveIt MoveGroup Python Interface Tutorial")
print("----------------------------------------------------------")
print("Press Ctrl-D to exit at any time")
print("")

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('plan', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

arm_group = moveit_commander.MoveGroupCommander("arm")

prev_pose.graspx = 0
prev_pose.graspy = 0
prev_pose.theta = 0

zero_pose.graspx = 0
zero_pose.graspy = 0
zero_pose.theta = 0

rate = rospy.Rate(20.0) # MUST be more then 2Hz

while 1:
    

    if next_pose_avail():

        print("\nIn Ifffffff")
        # print(prev_pose)
        print(new_pose)
        #Put the arm in the start position

        qx, qy, qz, qw = get_quaternion_from_euler(0,0,new_pose.theta)
        
        # arm_group.set_named_target("idle")
        # plan1 = arm_group.go()

        # arm_group.set_named_target("pick")
        # plan2 = arm_group.go()
        # time.sleep(1)

        # pose_target = geometry_msgs.msg.Pose()
        # pose_target.position.x = (new_pose.graspx)/1000
        # pose_target.position.y = (new_pose.graspy)/1000
        # print(pose_target.position.x)
        # print(pose_target.position.y)
        # pose_target.position.z = 0.09900598524181596
        # pose_target.orientation.x = -9.261859355487332e-06
        # pose_target.orientation.y = 0.7070801299313123
        # pose_target.orientation.z = 9.25930116420452e-06
        # pose_target.orientation.w = 0.7071334313160438
        # arm_group.set_pose_target(pose_target)
        # plan2 = arm_group.go()

        # time.sleep(1)

        # arm_group.set_named_target("drop")
        # plan3 = arm_group.go()

        # time.sleep(1)

        # arm_group.set_named_target("idle")
        # plan4 = arm_group.go()

        # put the arm at the 1st grasping position
        pose_target = geometry_msgs.msg.Pose()

        pose_target.position.x = (new_pose.graspy)/1000
        pose_target.position.y = -(new_pose.graspx)/1000
        pose_target.position.z = 0.10900598524181596
        pose_target.orientation.x = -9.261859355487332e-06
        pose_target.orientation.y = 0.7070801299313123
        pose_target.orientation.z = 9.25930116420452e-06
        pose_target.orientation.w = 0.7071334313160438
        arm_group.set_pose_target(pose_target)
        plan = arm_group.go()

        print('Above object')

        time.sleep(2)


        pose_target.position.x = (new_pose.graspy)/1000
        pose_target.position.y = -(new_pose.graspx)/1000
        pose_target.position.z = 0.05300598524181596
        pose_target.orientation.x = -9.261859355487332e-06
        pose_target.orientation.y = 0.7070801299313123
        pose_target.orientation.z = 9.25930116420452e-06
        pose_target.orientation.w = 0.7071334313160438
        arm_group.set_pose_target(pose_target)
        plan2 = arm_group.go()

        new_pose.width = 17
        local_pos_pub.publish(new_pose)

        print('Object picked')

        pose_target.position.x = (new_pose.graspy)/1000
        pose_target.position.y = -(new_pose.graspx)/1000
        pose_target.position.z = 0.10900598524181596
        pose_target.orientation.x = -9.261859355487332e-06
        pose_target.orientation.y = 0.7070801299313123
        pose_target.orientation.z = 9.25930116420452e-06
        pose_target.orientation.w = 0.7071334313160438
        arm_group.set_pose_target(pose_target)
        plan = arm_group.go()

        print('Above object')

        arm_group.set_named_target("drop")
        plan3 = arm_group.go()

        print('Drop reached')
        new_pose.width = 40
        local_pos_pub.publish(new_pose)

        arm_group.set_named_target("idle")
        plan4 = arm_group.go()

        print('Idle reached')
        rate.sleep()

    prev_pose = new_pose



group_variable_values = arm_group.get_current_joint_values()
print(group_variable_values)



