cmake_minimum_required(VERSION 3.0.0)
project(arm_description)

find_package(catkin REQUIRED COMPONENTS
	rospy
	roscpp
	geometry_msgs
	moveit_ros_planning_interface
)

# catkin_simple(ALL_DEPS_REQUIRED)


# cs_install()
# cs_export()

# cs_add_executable(plan_node scripts/plann.cpp)
# target_link_libraries(plan__node ${PROJECT_NAME})

# add_executable(plan_node scripts/plann.cpp)
# target_link_libraries(plan_node ${PROJECT_NAME} ${catkin_LIBRARIES})

# include_directories(include ${catkin_INCLUDE_DIRS})