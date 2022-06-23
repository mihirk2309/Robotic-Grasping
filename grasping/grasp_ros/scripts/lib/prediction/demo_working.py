# functionality for Camera calibration


from __future__ import absolute_import
import sys
sys.path.insert(0, '/home/mihir/catkin_ws/src/grasping/grasp_ros/scripts/lib')
Model_path = '/home/mihir/catkin_ws/src/grasping/grasp_ros/src_model/grasping_coep/models'
Model_name = 'model_29_Single_GPU.ckpt'
dataset_name = 'grasp'
dataset_path = '/home/mihir/catkin_ws/src/grasping/grasp_ros/src_model/grasping_coep/dataset/grasp'
image_set = 'test'

from calendar import c
from distutils.log import debug
import os

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

from  graspingnetwork.grasp_dataset import  GraspDataset
from  graspingnetwork.network import GraspNet
from  graspingnetwork.available_gpus import get_available_gpus

# For our kinect (input from grasping_node.py 550x550 image resized to 224x224 & then remapped back to 550x550) 
w_h_pixels = 550

gpus =0




# gpus = get_available_gpus(mem_lim=1024)
# if len(gpus):
#     if len(gpus) == 1:
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
#     else:
#         gpu_ids_str = ",".join(gpus)
#         print("gpus_str: {}".format(gpu_ids_str))
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str




ar = np.array

def load_model(gpu_ids):
    global device 
    #num_gpus = len(gpu_ids)  
    num_gpus = 0   
    model = GraspNet()
    
    # For old model.ckpt
    # state_dict = torch.load('./models/model.ckpt', map_location=lambda storage, loc: storage)
    # model.load_state_dict(state_dict)


    
    # # https://pytorch.org/tutorials/beginner/saving_loading_models.html



    DataParallel_used = 0  # if multiple gpu used

    if DataParallel_used ==1:
        # checkpoint = torch.load('./models/model_99.ckpt', map_location=lambda storage, loc: storage)
        # checkpoint.key()
        # state_dict = checkpoint['state_dict']
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove 'module.' of dataparallel
        #     new_state_dict[name]=v
        # model.load_state_dict(new_state_dict)

        checkpoint = torch.load('../models/model_29_Single_GPU.ckpt',  map_location=lambda storage,loc: storage)['model']
        # print(checkpoint.keys())
        # for key in list(checkpoint.keys()):
        #     if 'model.' in key:
        #         checkpoint[key.replace('model.', '')] = checkpoint[key]
        #         del checkpoint[key]
        # print(checkpoint.keys())
        model.load_state_dict(checkpoint, strict=False)


    else: 
        checkpoint = torch.load(os.path.join(Model_path,Model_name), map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])

    
    if num_gpus > 1:
        device = torch.device('cuda')
        model = nn.DataParallel(model).to(device)
    else: 
        if num_gpus == 1:
            device = torch.device("cuda:{}".format(gpu_ids[0]))
        else:
            device = torch.device("cpu")
        model.to(device)
    
    model.eval()
    return model


model = load_model(gpus)


batch_size = 1
dataset = GraspDataset(dataset_name, image_set, dataset_path)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Apply this transform for just only one image
inv_normalize = transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])



def Rotate2D(pts,cnt,ang=np.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts-cnt,ar([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]]))+cnt


# ground truth: green   predicted:red
def vis_detections(ax, im, score, dets, bbox_type):
    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    bbox = dets
    score = score

    # plot rotated rectangles
    pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
    cnt = ar([(int(bbox[0]) + int(bbox[2]))/2, (int(bbox[1]) + int(bbox[3]))/2])

    angle = score
    r_bbox = Rotate2D(pts, cnt, -np.pi/2-np.pi/20*(angle-1))
    pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
    pred_x, pred_y = pred_label_polygon.exterior.xy

    print("\n",bbox_type)
    print("pts:\n" , r_bbox)
    print("cnt:\n" , cnt)
    print("Angle:\n",angle*9)


    if bbox_type == 'Prediction':
        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
    elif bbox_type == 'Ground_truth':
        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='g', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='g', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)

def map(x, in_min, in_max, out_min, out_max):

   return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;


def get_grasp_config(im, score, dets):

    bbox = dets
    score = score

    # plot rotated rectangles
    pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
    cnt = ar([(int(bbox[0]) + int(bbox[2]))/2, (int(bbox[1]) + int(bbox[3]))/2])

    angle = score
    r_bbox = Rotate2D(pts, cnt, -np.pi/2-np.pi/20*(angle-1))
    pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
    pred_x, pred_y = pred_label_polygon.exterior.xy

    
    cnt_x = cnt[0]
    cnt_y = cnt[1]
    width =   ((((r_bbox[0][0] - r_bbox[1][0] )**2) + ((r_bbox[0][1]-r_bbox[1][1])**2) )**0.5)          # rematining
    theta = angle*9

    if(1):
        print("Rotated BBOX")
        print("pts:\n" , r_bbox)
        print("cnt:\n" , cnt)
        print("Angle:\n",angle*9)
        print("Width:\n",width)


    return cnt_x, cnt_y, width,theta


def get_Realworld_coordinates(cnt_x, cnt_y, width):
    # image_crop = image[0:700, 300:1420]
    # plt.imshow(image_crop)
    # plt.show()

    x_workbench_dist_mm, y_workbench_dist_mm = 1100, 675
    x_camera_pixel,y_camera_pixel = 1120, 685
    
    mm_per_pixelX = x_workbench_dist_mm/x_camera_pixel 
    mm_per_pixelY = y_workbench_dist_mm/y_camera_pixel

    cnt_x_mm, cnt_y_mm, width_mm = cnt_x*mm_per_pixelX, cnt_y*mm_per_pixelY, width*(mm_per_pixelX+mm_per_pixelY)/2

    cnt_x_mm = map(cnt_x_mm, 0,224, 0, w_h_pixels)
    cnt_y_mm = map(cnt_y_mm, 0,224, 0, w_h_pixels)
    width_mm = map(width_mm, 0,224, 0, w_h_pixels)


    return cnt_x_mm, cnt_y_mm, width_mm


def get_Realworld_coordinates_wrtBaselink(cnt_x_mm, cnt_y_mm, width_mm):

    TrasnformMatrix = np.array([[-1, 0, 338],
                            [0, 1, 120],
                            [0, 0 , 1]]
                            )
    Result = TrasnformMatrix.dot(np.array([cnt_x_mm, cnt_y_mm, 1]))

    cnt_x_mm = Result[0]  
    cnt_y_mm = Result[1]  

    return cnt_x_mm, cnt_y_mm, width_mm


def cv2_mirror(img):
    mirror_image = cv2.flip(img,1)
    return mirror_image


def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


def my_single_prediction(method_access, item_id=0):
    '''
      1-
      2-Test image accessing using index 
      3-single path (Real world image whichis saved)
      4-Test image accessing using path

    
    '''

    with torch.no_grad():        
            # Retrieve item
       # for index in range(1):
            # item = dataset[index*15]
            # img = item[0]
            # gt_rect = item[1]
            # print("\n", type(img),"\n")

        
        if(method_access == 1):
            img = io.imread('/home/pranay/robotic-grasping-cornell/dataset/grasp/Images/pcd0116r_preprocessed_4.png')
            print(img.shape)
            print(type(img))
            
        elif(method_access == 2):     # for reading test images from dataset from dataset by taking ID of the test image
            # item = dataset[0]
            # img = item[0]
            # gt_rect = item[1]
            # print("\n We at Method 2: ", type(img),"\n")
            # gth_cls = torch.tensor(gt_rect[0])
            # gth_rect = torch.tensor(gt_rect[1])
            # print("\n this is gt_rect:",  gth_cls, gth_rect )
           
            # name = 'grasp'
            # dataset_path = './dataset/grasp'
            # image_set = 'train'
            # train_dataset = GraspDataset(name, image_set, dataset_path)
            
            img, gt_rect = dataset.__getitem__(int(item_id))


            # CxHxW -> HxWxC
            # img = np.transpose(img,(1,2,0))
            gth_cls = gt_rect[0]
            gth_rect = gt_rect[1]
            print('gt_cls: {0},\n gt_rect: {1}'.format(gt_rect[0], gt_rect[1]))
            print('gt_cls: {0},\n gt_rect: {1}'.format( gth_cls,gth_rect))
            
            if 0:
                img = np.transpose(img,(1,2,0))
                plt.imshow(img)
                plt.show()
            
            gth_cls= gth_cls.squeeze()
            gth_cls =gth_cls.cpu()
            gth_cls = gth_cls.detach().numpy()
            gth_cls= gth_cls.squeeze()

            #gth_cls = np.argmax(gth_cls)

            gth_rect = gth_rect.cpu()
            gth_rect = gth_rect.detach().numpy()

            print(img.shape)
            print(type(img))
        
        elif(method_access == 3):     # this is used to read saved image and pass it to the model
            img = io.imread('/home/mihir/catkin_ws/undertest_img_rgd')

            if 0: 
                cv2.imshow("Input  to model", img)
                cv2.waitKey(0)
                

            img = cv2.resize(img, (224,224))

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([transforms.ToTensor(), normalize]) 
            img = transform(img)


        else:                   # for testing from dataset
            debug_demo = 0
            img = io.imread('/home/pranay/catkin_ws/src/grasp_ros/src_model/grasping_coep/dataset/grasp/Images_Test/Cropped320_rgd/pcd0102r_preprocessed_1.png')
            
            if debug_demo == 1:
                plt.subplot(2,1,1)
                plt.imshow(img)

            img = center_crop(img,300,300)
            img = cv2.resize(img, (224,224))

         

            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([transforms.ToTensor(), normalize]) 
            img = transform(img)

            if debug_demo == 1:
                plt.subplot(2,1,2)
                img = img.permute(1, 2, 0)   # Given a Tensor representing the image, use .permute() to put the channels as the last dimension
                plt.imshow(img)
                plt.show()




        
        img = img.to(device)
        rect_pred, cls_score = model(img[None, ...])     # pass image to the model

        cls_score = cls_score.squeeze()
        rect_pred = rect_pred.squeeze()
        cls_score = cls_score.cpu()
        cls_score = cls_score.detach().numpy()
        ind_max = np.argmax(cls_score)        
        rect_pred = rect_pred.cpu()
        rect_pred = rect_pred.detach().numpy()

        img = inv_normalize(img)
        img = img.numpy()
        img = np.transpose(img,(1,2,0))




        print('\nrect_pred: {0}'.format(rect_pred))
        print("index:",ind_max ,"  angle:", ind_max*9)

        # CALL get_grasp_config() to get all the predicted parameter(x,y,width,theta)
        cnt_x, cnt_y, width,theta = get_grasp_config(img, ind_max, rect_pred)
        print("\nGrasp Congig: ",    cnt_x, cnt_y, width,theta )
       
        cnt_x_mm, cnt_y_mm, width_mm = get_Realworld_coordinates(cnt_x, cnt_y, width)    
        print("Realworld Results (mm_x,mm_y,mm_width, theta): ", cnt_x_mm, cnt_y_mm, width_mm,theta  )


        cnt_x_mm, cnt_y_mm, width_mm =  get_Realworld_coordinates_wrtBaselink(cnt_x_mm, cnt_y_mm, width_mm)
        print("Results wrt Baselink in mm: ", cnt_x_mm, cnt_y_mm, width_mm  )



       
       # Create figure and axes
        fig,ax = plt.subplots(1)
        ax.imshow(img)
        vis_detections(ax, img, ind_max, rect_pred,"Prediction")      # predicted
        plt.draw()
        
        if method_access == 2:
            vis_detections(ax, img, gth_cls, gth_rect,"Ground_truth")      # ground truth
            plt.draw()
            print("\n  gth_cls, gth_rect: ", gth_cls, gth_rect,"\n",type(gth_cls),type(gth_rect))
        
        plt.show(block=False)
        plt.pause(10)
        plt.close()

        return cnt_x_mm, cnt_y_mm, width_mm,theta










##############################################################################


        # Below is the not so important code #

#############################################################################


def get_M_CL_info(gray, image_init, visualize=False):
    # parameters
    markerLength_CL = 0.093
    aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    # aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    corners_CL, ids_CL, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict_CL, parameters=parameters)

    # for the first frame, it may contain nothing
    if ids_CL is None:
        return default_M_CL, None

    rvec_CL, tvec_CL, _objPoints_CL = aruco.estimatePoseSingleMarkers(corners_CL[0], markerLength_CL,
                                                                      cameraMatrix, distCoeffs)
    dst_CL, jacobian_CL = cv2.Rodrigues(rvec_CL)
    M_CL = np.zeros((4, 4))
    M_CL[:3, :3] = dst_CL
    M_CL[:3, 3] = tvec_CL
    M_CL[3, :] = np.array([0, 0, 0, 1])

    if visualize:
        # print('aruco is located at mean position (%d, %d)' %(mean_x ,mean_y))
        aruco.drawAxis(image_init, cameraMatrix, distCoeffs, rvec_CL, tvec_CL, markerLength_CL)
    return M_CL, corners_CL[0][0, :, :]


def aruco_tag_remove(rgb_image, corners):
    img_out = rgb_image.copy()

    # find the top-left and right-bottom corners
    min = sys.maxsize
    max = -sys.maxsize
    tl_pxl = None
    br_pxl = None
    for corner in corners:
        if corner[0] + corner[1] < min:
            min = corner[0] + corner[1]
            tl_pxl = [int(corner[0]), int(corner[1])]

        if corner[0] + corner[1] > max:
            max = corner[0] + corner[1]
            br_pxl = [int(corner[0]), int(corner[1])]

    # get the replacement pixel value
    rep_color = img_out[tl_pxl[0] - 10, tl_pxl[1] - 10, :]

    for h in range(tl_pxl[1] - 45, br_pxl[1] + 46):
        for w in range(tl_pxl[0] - 45, br_pxl[0] + 46):
            img_out[h, w, :] = rep_color

    return img_out

def project(pixel, depth_image, M_CL, M_BL, cameraMatrix):
    '''
     project 2d pixel on the image to 3d by depth info
     :param pixel: x, y
     :param M_CL: trans from camera to aruco tag
     :param cameraMatrix: camera intrinsic matrix
     :param depth_image: depth image
     :param depth_scale: depth scale that trans raw data to mm
     :return:
     q_B: 3d coordinate of pixel with respect to base frame
     '''
    depth = depth_image[pixel[1], pixel[0]]

    # if the depth of the detected pixel is 0, check the depth of its neighbors
    # by counter-clock wise
    nei_range = 1
    while depth == 0:
        for delta_x in range(-nei_range, nei_range + 1):
            for delta_y in range(-nei_range, nei_range + 1):
                nei = [pixel[0] + delta_x, pixel[1] + delta_y]
                depth = depth_image[nei[1], nei[0]]

                if depth != 0:
                    break

            if depth != 0:
                break

        nei_range += 1

    pxl = np.linalg.inv(cameraMatrix).dot(np.array([pixel[0] * depth, pixel[1] * depth, depth]))
    q_C = np.array([pxl[0], pxl[1], pxl[2], 1])
    q_L = np.linalg.inv(M_CL).dot(q_C)
    q_B = M_BL.dot(q_L)

    return q_B

def pre_process(rgb_img, depth_img):
    inp_image = rgb_img
    inp_image[:, :, 0] = depth_img

    inp_image = cv2.resize(inp_image, (256, 256))

    return inp_image

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

        cv2.imshow("Depth", cv_depth)
        cv2.imshow("RGB", cv_rgb)

        img = cv_rgb_arr.copy()
        depth_raw = cv_depth_arr.copy()

        gray = img.astype(np.uint8)
        depth = (depth_raw * 1000).astype(np.uint8)

        # get the current transformation from the camera to aruco tag
        M_CL, corners = get_M_CL_info(gray, img, False)

        # remove aruco tag from input image to avoid mis-detection
        if corners is not None:
            img_wo_at = aruco_tag_remove(img, corners)

        # replace blue channel with the depth channel
        inp_image = pre_process(img_wo_at, depth)

        # pass the image into the network
        ret = detector.run(inp_image[:, :, :])
        ret = ret["results"]

        loc_ori = KpsToGrasppose(ret, img, depth_raw, M_CL, M_BL, cameraMatrix)
        pub_res.publish(loc_ori)

    except CvBridgeError as e:
        print(e)

def isWithinRange(pxl, w, h):
    x, y = pxl[:]

    return w/12. <= x <= 11*w/12 and h/12. <= y <= 11*h/12

def KpsToGrasppose(net_output, rgb_img, depth_map, M_CL, M_BL, cameraMatrix, visualize=True):
    kps_pr = []
    for category_id, preds in net_output.items():
        if len(preds) == 0:
            continue

        for pred in preds:
            kps = pred[:4]
            score = pred[-1]
            kps_pr.append([kps[0], kps[1], kps[2], kps[3], score])

    # no detection
    if len(kps_pr) == 0:
        return [0, 0, 0, 0]

    # sort by the confidence score
    kps_pr = sorted(kps_pr, key=lambda x: x[-1], reverse=True)
    # select the top 1 grasp prediction within the workspace
    res = None
    for kp_pr in kps_pr:
        f_w, f_h = 640. / 512., 480. / 512.
        kp_lm = (int(kp_pr[0] * f_w), int(kp_pr[1] * f_h))
        kp_rm = (int(kp_pr[2] * f_w), int(kp_pr[3] * f_h))

        if isWithinRange(kp_lm, 640, 480) and isWithinRange(kp_rm, 640, 480):
            res = kp_pr
            break

    if res is None:
        return [0, 0, 0, 0]

    f_w, f_h = 640./512., 480./512.
    kp_lm = (int(res[0]*f_w), int(res[1]*f_h))
    kp_rm = (int(res[2]*f_w), int(res[3]*f_h))
    center = (int((kp_lm[0]+kp_rm[0])/2), int((kp_lm[1]+kp_rm[1])/2))

    kp_lm_3d = project(kp_lm, depth_map, M_CL, M_BL, cameraMatrix)
    kp_rm_3d = project(kp_rm, depth_map, M_CL, M_BL, cameraMatrix)
    center_3d = project(center, depth_map, M_CL, M_BL, cameraMatrix)

    orientation = np.arctan2(kp_rm_3d[1] - kp_lm_3d[1], kp_rm_3d[0] - kp_lm_3d[0])
    # motor 7 is clockwise
    if orientation > np.pi / 2:
        orientation = np.pi - orientation
    elif orientation < -np.pi / 2:
        orientation = -np.pi - orientation
    else:
        orientation = -orientation

    # compute the open width
    dist = np.linalg.norm(kp_lm_3d[:2] - kp_rm_3d[:2])

    # draw arrow for left-middle and right-middle key-points
    lm_ep = (int(kp_lm[0] + (kp_rm[0] - kp_lm[0]) / 5.), int(kp_lm[1] + (kp_rm[1] - kp_lm[1]) / 5.))
    rm_ep = (int(kp_rm[0] + (kp_lm[0] - kp_rm[0]) / 5.), int(kp_rm[1] + (kp_lm[1] - kp_rm[1]) / 5.))
    rgb_img = cv2.arrowedLine(rgb_img, kp_lm, lm_ep, (0, 0, 0), 2)
    rgb_img = cv2.arrowedLine(rgb_img, kp_rm, rm_ep, (0, 0, 0), 2)
    # draw left-middle, right-middle and center key-points
    rgb_img = cv2.circle(rgb_img, (int(kp_lm[0]), int(kp_lm[1])), 2, (0, 0, 255), 2)
    rgb_img = cv2.circle(rgb_img, (int(kp_rm[0]), int(kp_rm[1])), 2, (0, 0, 255), 2)
    rgb_img = cv2.circle(rgb_img, (int(center[0]), int(center[1])), 2, (0, 0, 255), 2)

    if visualize:
        cv2.namedWindow('visual', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('visual', rgb_img)

    return [center_3d[0], center_3d[1], center_3d[2], orientation, dist]




#main code

# Projection matrix of Kinect
P = np.array([[1081.372, 0.0, 959.5],
              [0.0, 1081.372, 539.5],
              [0.0, 0.0, 1.0]]
            )

# camera intrinsic matrix of Kinect v2
cameraMatrix = np.array([[1081.372, 0.0, 959.5],
                        [0.0, 1081.372, 539.5],
                        [0.0, 0.0, 1.0]]
                )

#np.linalg.inv(cameraMatrix).dot(np.array[2,2,2])


if __name__ == '__main__':

    

    print("Length of test_loader:",len(test_loader.dataset))


    # my_single_prediction(3)   # 3 is real world implementation
    # quit()

    # for method access 2
    for id in range(177):
        #id = input("\nEnter the id of image to Predict: ")
        my_single_prediction(2,id)


# .........................................................................
#  BELOW CODE NOT USED


    i=0
    if i==1:
    #with torch.no_grad():
        my_single_prediction()
        for i, (img, gt_rect) in enumerate(test_loader):
            
            img = img.to(device)
            #print('img.size(): {}'.format(img.size()))
            rect_pred, cls_score = model(img)   
            cls_score = cls_score.squeeze()
            rect_pred = rect_pred.squeeze()
            #print('cls_score.shape: {}'.format(cls_score.shape))
            cls_prob = F.softmax(cls_score,0)
            #print('cls_prob: {0}'.format(cls_prob))
            #print('rect_pred: {0}'.format(rect_pred))
            
            img = img.cpu()
            img = img[0,:,:,:]
            img = inv_normalize(img)
            img = img.numpy()
            # CxHxW -> HxWxC
            img = np.transpose(img,(1,2,0))
            
            cls_score = cls_score.cpu()
            cls_score = cls_score.detach().numpy()
            ind_max = np.argmax(cls_score)
            #print('ind_max: {}, cls_score[{}]: {}'.format(ind_max, ind_max, cls_score[ind_max]))
            
            rect_pred = rect_pred.cpu()
            rect_pred = rect_pred.detach().numpy()
            print('rect_pred: {0}'.format(rect_pred))
            print("ind:",ind_max ,"ang", ind_max*9)
            
            p1 = (rect_pred[0], rect_pred[1])
            p2 = (rect_pred[0] + rect_pred[2], rect_pred[1])
            p3 = (rect_pred[0] + rect_pred[2], rect_pred[1] + rect_pred[3])
            p4 = (rect_pred[0], rect_pred[1] + rect_pred[3])

            # Create figure and axes
            fig,ax = plt.subplots(1)
            # Display the image
            ax.imshow(img)
            vis_detections(ax, img, ind_max, rect_pred)
            plt.draw()
            plt.show()
        
        
            '''
            cv2.line(img, p1, p2, (0, 0, 255), 2)
            cv2.line(img, p2, p3, (0, 0, 255), 2)
            cv2.line(img, p3, p4, (0, 0, 255), 2)
            cv2.line(img, p4, p1, (0, 0, 255), 2)
            cv2.imshow('bbox', img)
            cv2.waitKey(0)
            '''
            
            if i > 3:
                break










