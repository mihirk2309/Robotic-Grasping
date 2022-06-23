#! /usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()    
group = moveit_commander.MoveGroupCommander("arm")
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory)

group_name = "arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

# group_variable_values = group.get_current_joint_values()

# group_variable_values[0] = 0.7
# group_variable_values[1] = -0.7
# group_variable_values[3] = 0
# #group_variable_values[5] = 1.5
# group.set_joint_value_target(group_variable_values)

# plan2 = group.plan()
# group.go(wait=True)

pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.x = 7.9971e-05
pose_goal.orientation.y = 0.70713
pose_goal.orientation.z = 4.2845e-05
pose_goal.orientation.w = 0.70708
pose_goal.position.x = 0.00556938
pose_goal.position.y = -0.124437
pose_goal.position.z = 0.197501
# move_group.set_goal_tolerance(0.001);
move_group.set_pose_target(pose_goal)

## Now, we call the planner to compute the plan and execute it.
plan = move_group.go(wait=True)

rospy.sleep(5)

moveit_commander.roscpp_shutdown()