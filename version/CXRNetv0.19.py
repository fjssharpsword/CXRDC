# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 28/12/2020
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
#from config import *
#construct model
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(ImageClassifier, self).__init__()
        self.msa = MultiScaleAttention()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        
    def forward(self, x):
        #x: N*C*W*H
        x = self.msa(x) * x
        global_fea = self.dense_net_121.features(x)
        global_fea = self.sigmoid(global_fea)
        out = self.avgpool(global_fea)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return global_fea, out
        
class MultiScaleAttention(nn.Module):#multi-scal attention module
    def __init__(self):
        super(MultiScaleAttention, self).__init__()
        
        self.scaleConv1 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False)
        self.scaleConv2 = nn.Conv2d(3, 3, kernel_size=9, padding=4, bias=False)
        
        self.aggConv = nn.Conv2d(6, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        out_avg = torch.mean(x, dim=1, keepdim=True)
        
        out1 = self.scaleConv1(x)
        out_max1, _ = torch.max(out1, dim=1, keepdim=True)
        out_avg1 = torch.mean(out1, dim=1, keepdim=True)
        
        out2 = self.scaleConv2(x)
        out_max2, _ = torch.max(out2, dim=1, keepdim=True)
        out_avg2 = torch.mean(out2, dim=1, keepdim=True)

        x = torch.cat([out_max, out_avg, out_max1, out_avg1, out_max2, out_avg2], dim=1)
        x = self.sigmoid(self.aggConv(x))

        return x

class RegionComparer(nn.Module):
    def __init__(self):
        super(RegionComparer, self).__init__()
        self.adapool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(49, 49) 
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, global_fea, mask):
        #x: N*C*W*H
        mask = self.adapool(mask)
        mask = mask.ge(0.5).float() #0,1 binarization
        #mask = mask.repeat(1, conv_fea.size(1), 1, 1) #bz*1*7*7 -> bz*1024*7*7
        """
        #show
        cam = conv_fea[0].detach().cpu().squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.applyColorMap(np.uint8(cam * 255.0), cv2.COLORMAP_JET) #L to RGB
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(cam)
        ax.axis('off')
        fig.savefig('/data/pycode/CXR-IRNet/imgs/cam_before.png')
        """
        region_fea = torch.mul(global_fea, mask)
        """
        #show
        cam = conv_fea[0].detach().cpu().squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.applyColorMap(np.uint8(cam * 255.0), cv2.COLORMAP_JET) #L to RGB
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(cam)
        ax.axis('off')
        fig.savefig('/data/pycode/CXR-IRNet/imgs/cam_after.png')
        """
        out = torch.mean(region_fea, dim=1, keepdim=True)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

#https://github.com/qianjinhao/circle-loss/blob/master/circle_loss.py
class CircleLoss(nn.Module):
    def __init__(self, scale=1, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        mask = torch.matmul(labels, torch.t(labels))
        mask = torch.where(mask==2, torch.ones_like(mask), mask) #for multi-label
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]
        #neg_pair_ = sim_mat[neg_mask == 1][0:len(pos_pair_)] #for sampling part normal 

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

if __name__ == "__main__":
    #for debug   
    img = torch.rand(10, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = ImageClassifier(num_classes=14, is_pre_trained=True)
    global_fea, out = model(img)
    print(out.size())


