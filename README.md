# Deep Learning based Robotic Grasping
[comment]: <>  (## _The Last Markdown Editor, Ever_)

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

### *Best Final Year Project Award at Electronics and Telecommunication Department, College of Engineering Pune (COEP)*. 

### Problem Statement: 
The problem statement consists of grasping unknown, unordered, and ran- domly oriented objects. Normally arms can pick objects that are initially placed in a predefined order for example an assembly line. The robot is hard- coded to pick and place known items. But what if the items are unknown or placed in a random manner or what if there are multiple objects stacked in a place and we want to separate items individually. This is where a CV-based algorithm is needed. The arm should automatically orient itself in a suitable grasping position which will be different for different objects and given by the algorithm. The two main points of problem statement that we wish to address are: 
- Grasping different types of objects having different shape and size. Ref. Figure 3.2 
- Grasping object which is in different pose(i.e. position and orientation). Ref. Figure 3.3

### Abstract:
The problem of robotic grasping is still an unsolved problem with many approaches trying to generalize grasp predictions for unseen and dynamic en- vironments. Here we explore two approaches, one based on transfer learning, and another using a popular grasp detection model known as GG-CNN. In the transfer learning approach we tried 2 base models, VGG-16 and ResNet- 50. ResNet-50 provided better results with a testing accuracy of 83.3% while VGG-16 provided an accuracy of 78.2%. In order to test our model on a real robotic arm, we built a 5-DOF arm and added a custom parallel plate gripper. Complete ROS and Moveit support is added to our developed robotic arm. The processed RG-D image from the KinectV2 camera is given as an input to the model which predicts the 5-D grasp configuration. Required electronic system design and its PCB is built which controls the robotic arm. The pre- dicted 5-D grasp configuration is then transformed to the object pose w.r.t the base link frame of the robot. Finally, A ROS node that automates the task of picking objects lying in different positions & orientations and sends the joint angle values over pyserial communication to the Arduino (PCB) is written. Thus, we have developed a complete pipeline for the task of Deep Learning based robotic grasping.

####  This repository contains following packages:
- Arduino : This repo contains Arduino Code for Controlling servo motors attached to each joints. 
- grasping : This repo contains the Grasping Model, Moveit setup, calibration and entire grasping pipeline code.
- iai_kinect2_opencv4 : Contains the packages for Kinect interfacing, calibration, ROS integration with kinect, etc.
- moveit_calibration: Contains Robotic arm calibration files in a hand-eye setup.

##### Main "grasping" folder structure: 
-  Models_and_dataset
This repo contains trained model and Dat
- arm_description
This pacakge contains URDF file of our 5-DOF Robotic arm. 
- arm1_moveit_config
This package contains Moveit config files for our Robotic arm.
- grasp_ros
This package contains all the scripts required for grasp prediction, visualising Kinect data, Planning arm trajectory.


## Codes:

### 1. Pre-processed  Dataset

+ Download RG-D dataset [Cornell Dataset](https://drive.google.com/file/d/1AW7le2PbktTAVgZ3RSeInOhw16xwO026/view?usp=sharing) 
+ Run `dataPreprocessingTest_fasterrcnn_split.m` in Matlab (please modify paths according to your structure) 

### 2. Training
Go to the repository containing train.py file.
```
$ python3 train.py --epochs 30 --lr 0.0001 --batch_size 8
```
### 3. Demo on our 5-DOF Robotic arm
##### Run following commands in separate Linux Terminal to test the model on our 5-DOF Robotic arm.

+ To start kinect sensor: ``` $ roslaunch kinect2_bridge kinect2_bridge.launch```
+ To start Moveit package: ``` $ roslaunch arm1_moveit_config demo.launch```
+ To send IK solution to arduino through Pyserial: ``` $ rosrun arm_description send_new.py ```
+ To start prediction: ``` $ rosrun grasp_ros grasping_node.py ```
+ To start grasping pipeline:  ``` $ rosrun arm_description task.py ```


### 4. Requirements
+ ROS Melodic
+ Python 3.7
+ Ubuntu 20.04 LTS

### 5. Working video

https://user-images.githubusercontent.com/63548541/175263594-90140005-540c-467b-b42f-4648c0e92eb1.mp4

# Detailed Project Report: 
https://drive.google.com/file/d/1E5o1nOsbKbwT3pMwrMMl_3RMYuTg9NtT/view?usp=sharing

## License

MIT

