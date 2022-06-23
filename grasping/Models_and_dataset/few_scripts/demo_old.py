# raw working
import os
from available_gpus import get_available_gpus


gpus =0

# gpus = get_available_gpus(mem_lim=1024)
# if len(gpus):
#     if len(gpus) == 1:
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
#     else:
#         gpu_ids_str = ",".join(gpus)
#         print("gpus_str: {}".format(gpu_ids_str))
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        
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
from shapely.geometry import Polygon

from grasp_dataset import GraspDataset
from network import GraspNet

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

        checkpoint = torch.load('../models/model_99.ckpt',  map_location=lambda storage,loc: storage)['model']
        # print(checkpoint.keys())
        # for key in list(checkpoint.keys()):
        #     if 'model.' in key:
        #         checkpoint[key.replace('model.', '')] = checkpoint[key]
        #         del checkpoint[key]
        # print(checkpoint.keys())
        model.load_state_dict(checkpoint, strict=False)


    else: 
        checkpoint = torch.load('../models/model_2.ckpt', map_location=lambda storage, loc: storage)
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
    print("pts:" , pts)
    print("cnt:" , cnt)
    angle = score
    r_bbox = Rotate2D(pts, cnt, -np.pi/2-np.pi/20*(angle-1))
    pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
    pred_x, pred_y = pred_label_polygon.exterior.xy

    if bbox_type == 'prediction':
        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
    elif bbox_type == 'ground_truth':
        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='g', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='g', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)


def cv2_mirror(img):
    mirror_image = cv2.flip(img,1)
    return mirror_image

def my_single_prediction(item_id):
    with torch.no_grad():        
            # Retrieve item
       # for index in range(1):
            # item = dataset[index*15]
            # img = item[0]
            # gt_rect = item[1]
            # print("\n", type(img),"\n")

        method_access = 2
        if(method_access == 1):
            img = io.imread('/home/pranay/robotic-grasping-cornell/dataset/grasp/Images/pcd0116r_preprocessed_4.png')
            print(img.shape)
            print(type(img))
            
        elif(method_access == 2):
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
            # plt.imshow(img)
            # plt.show()
            gth_cls= gth_cls.squeeze()
            gth_cls =gth_cls.cpu()
            gth_cls = gth_cls.detach().numpy()
            gth_cls= gth_cls.squeeze()
            #gth_cls = np.argmax(gth_cls)

            gth_rect = gth_rect.cpu()
            gth_rect = gth_rect.detach().numpy()

            print(img.shape)
            print(type(img))
        


        else:
            img = io.imread('/home/pranay/robotic-grasping-cornell/test_images/test_stripper/rgb_marker.png')
            img = cv2_mirror(img)
            print(img.shape)
            #cv2.imshow("test", img)
            img = img[0:700 ,0:700]   #y,y+h   x,x+w
            print(img.shape)
            #cv2.imshow("test_crop", img)
            img = cv2.resize(img, (224,224))
            # print(img.shape)
            # print(type(img))
            #cv2.imshow("test_resize", img)
            #cv2.waitKey(0)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([transforms.ToTensor(), normalize]) 
            img = transform(img)
            print(img.shape)
            print(type(img))


        
        print("\n", type(img),"\n")
        img = img.to(device)
        rect_pred, cls_score = model(img[None, ...])   
#        rect_pred, cls_score = model(img)   
        cls_score = cls_score.squeeze()
        rect_pred = rect_pred.squeeze()
        print("\ntype image", type(img), img.size() )
        # cls_prob = F.softmax(cls_score,0)
        # img = img.cpu()
        # img = img[0,:,:,:]
        img = inv_normalize(img)
        img = img.numpy()
        img = np.transpose(img,(1,2,0))
        cls_score = cls_score.cpu()
        cls_score = cls_score.detach().numpy()
        ind_max = np.argmax(cls_score)        
        rect_pred = rect_pred.cpu()
        rect_pred = rect_pred.detach().numpy()
        print('rect_pred: {0}'.format(rect_pred))
        print("ind:",ind_max ,"ang", ind_max*9)

        cls_score = cls_score.squeeze()
            # Create figure and axes
        fig,ax = plt.subplots(1)
            # Display the image
        ax.imshow(img)
        vis_detections(ax, img, ind_max, rect_pred,"prediction")
        plt.draw()
        print("\n  ind_max, rect_pred: ", ind_max,rect_pred, "\n",type(ind_max),type(rect_pred))
        print("\n  gth_cls, gth_rect: ", gth_cls, gth_rect,"\n",type(gth_cls),type(gth_rect))
        vis_detections(ax, img, gth_cls, gth_rect,"ground_truth") 
        plt.draw()
        plt.show(block=False)
        plt.pause(3)
        plt.close()


#main code

model = load_model(gpus)
dataset_name = 'grasp'
dataset_path = './dataset/grasp'
image_set = 'test'
# Apply this transform for just only one image
inv_normalize = transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])

batch_size = 1
dataset = GraspDataset(dataset_name, image_set, dataset_path)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print("Length of test_loader:",len(test_loader.dataset))
# to visulaise both ground truth and predicted boundin boxes
for i in range(10):
    id = input("\nEnter the id of image to Predict: ")
    my_single_prediction(id)



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










