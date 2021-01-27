# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 20/01/2021
"""
import sys
import re
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label as skmlabel
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
#define myself
from config import *

class CXRClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(CXRClassifier, self).__init__()
        self.sa_left = SpatialAttention()
        self.sa_right = SpatialAttention()
        self.sa_heart = SpatialAttention()
        self.sa_global = SpatialAttention()
        self.dense_net_121_local = torchvision.models.densenet121(pretrained=is_pre_trained)
        self.dense_net_121_global = torchvision.models.densenet121(pretrained=is_pre_trained)
        self.pool_local = nn.AvgPool2d(kernel_size=7, stride=1)
        self.pool_global = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier_local = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        self.classifier_global = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        
    def forward(self, x, is_patch = True):
        if is_patch==True:#local learnging, x=NX3*CXWXH
            x = torch.cat( [self.sa_left(x[:,0:3,:,:]), self.sa_right(x[:,3:6,:,:]), self.sa_heart(x[:,6:,:,:])], dim=1) 
            x = self.dense_net_121_local.features(x)
            x = self.pool_local(x).view(x.size(0), -1)
            x = self.classifier_local(x)
            """
            cam = x[0].sum(0).data.cpu().squeeze().numpy()#1024*7*7
            cam = (cam - cam.min()) / (cam.max() - cam.min()) #[0,1]
            cam = cv2.applyColorMap(np.uint8(cam* 255.0) , cv2.COLORMAP_JET) #L to RGB
            fig, ax = plt.subplots(1)# Create figure and axes
            ax.imshow(cam)
            ax.axis('off')
            fig.savefig('/data/pycode/CXR-IRNet/imgs/oriImage_attention_map_local.png')
            """
            
            """
            cam = image[0].sum(0).data.cpu().squeeze().numpy()#1024*7*7
            cam = (cam - cam.min()) / (cam.max() - cam.min()) #[0,1]
            cam = cv2.applyColorMap(np.uint8(cam* 255.0) , cv2.COLORMAP_JET) #L to RGB
            fig, ax = plt.subplots(1)# Create figure and axes
            ax.imshow(cam)
            ax.axis('off')
            fig.savefig('/data/pycode/CXR-IRNet/imgs/oriImage_attention_map_global.png')
            """
        else:#global learning, x=NXCXWXH
            x = self.sa_global(x) * x
            x = self.dense_net_121_global.features(x)
            #x = torch.mul(x, image)
            x = self.pool_global(x).view(x.size(0), -1)
            x = self.classifier_global(x)

        return x
        
class SpatialAttention(nn.Module):#spatial attention layer
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def ROIGeneration(images, masks_list, labels):
    patchs = torch.FloatTensor()
    patch_labels = torch.FloatTensor()
    globals = torch.FloatTensor()
    global_labels = torch.FloatTensor()
    for i in range(0, images.size(0)):
        image = images[i] #3*224*224
        label = labels[i]
        patchs_local = torch.FloatTensor()
        for masks in masks_list: 
            mask = masks[i].data.cpu().squeeze() #224*224
            mask = mask.ge(0.5).float().numpy() #0,1 binarization
            """
            mask = masks[i].data.cpu().repeat(image.size(0), 1, 1)
            image = torch.mul(image, mask)
            cam = image.permute(2, 1, 0).numpy() #224*224*3
            cam = (cam - cam.min()) / (cam.max() - cam.min()) #[0,1]
            cam = np.uint8(cam* 255.0) 
            fig, ax = plt.subplots(1)# Create figure and axes
            ax.imshow(cam)
            ax.axis('off')
            fig.savefig('/data/pycode/CXR-IRNet/imgs/oriImage_seg_heart.png')
            """
            ind = np.argwhere(mask != 0)
            if len(np.unique(ind[:,0]))>2 and len(np.unique(ind[:,1]))>2:
                minh = min(ind[:,0])
                minw = min(ind[:,1])
                maxh = max(ind[:,0])
                maxw = max(ind[:,1])
                image_crop = image.permute(1,2,0).squeeze().numpy() #224*224*3
                image_crop = image_crop[minh:maxh,minw:maxw,:]
                image_crop = cv2.resize(image_crop, (config['TRAN_CROP'],config['TRAN_CROP']))
                """
                cam = image_crop #224*224*3
                cam = (cam - cam.min()) / (cam.max() - cam.min()) #[0,1]
                cam = np.uint8(cam* 255.0) 
                fig, ax = plt.subplots(1)# Create figure and axes
                ax.imshow(cam)
                ax.axis('off')
                fig.savefig('/data/pycode/CXR-IRNet/imgs/oriImage_crop_heart.png')
                """
                image_crop = torch.FloatTensor(image_crop).permute(2, 1, 0) #3*224*224
                patchs_local = torch.cat((patchs_local, image_crop), 0)
        if patchs_local.size(0) ==9:
            patchs = torch.cat((patchs, patchs_local.unsqueeze(0)), 0)
            patch_labels = torch.cat((patch_labels, label.unsqueeze(0)), 0)
        else:
            globals = torch.cat((globals, image.unsqueeze(0)), 0)
            global_labels = torch.cat((global_labels, label.unsqueeze(0)), 0)
    return patchs, patch_labels, globals, global_labels

if __name__ == "__main__":
    #for debug   
    x = torch.rand(32, 9, 224, 224)
    model = CXRClassifier(num_classes=14, is_pre_trained=True)
    out = model(x)
    print(x.size())

