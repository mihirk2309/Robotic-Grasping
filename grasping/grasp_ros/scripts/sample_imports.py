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



test_working =1
if(test_working ==1):
    for id in range(177):
        print("\nGRASPING DETECTION")
        cnt_x_mm, cnt_y_mm, width_mm,theta = demo_working.my_single_prediction(2,id)   #show prediction on all images 
        print("\n\nOUTSIDE MAIN:",cnt_x_mm, cnt_y_mm, width_mm,theta )