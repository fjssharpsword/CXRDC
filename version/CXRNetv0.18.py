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
from config import *
#construct model
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(ImageClassifier, self).__init__()
        self.msa = MultiScaleAttention()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Sequential(nn.Linear(49, num_classes), nn.Sigmoid())
        
    def forward(self, x):
        #x: N*C*W*H
        x = self.msa(x) * x
        conv_fea = self.dense_net_121.features(x)
        conv_fea = conv_fea.mean(1).unsqueeze(1) #1024*7*7->1*7*7
        fc_fea = conv_fea.view(conv_fea.size(0), -1)
        #fc_fea = self.avgpool(conv_fea).view(conv_fea.size(0), -1)
        out = self.classifier(fc_fea)
        return conv_fea, fc_fea, out
        
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

class RegionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RegionClassifier, self).__init__()
        self.adapool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(49, num_classes), nn.Sigmoid())
        
    def forward(self, conv_fea, mask):
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
        conv_fea = torch.mul(conv_fea, mask)
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
        fc_fea = conv_fea.view(conv_fea.size(0), -1)
        out = self.classifier(fc_fea)
        return fc_fea, out

class FusionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FusionClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_size, num_classes), nn.Sigmoid())

    def forward(self, fc_fea_img, fc_fea_roi):
        fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
        out = self.classifier(fc_fea_fusion)
        return out

if __name__ == "__main__":
    #for debug   
    img = torch.rand(32, 3, 224, 224).to(torch.device('cuda:%d'%7))
    label = torch.zeros(32, 14)
    for i in range(32):#generate 1 randomly
        ones_n = random.randint(1,2)
        col = [random.randint(0,13) for _ in range(ones_n)]
        label[i, col] = 1
    model_img = CXRClassifier(num_classes=14, is_pre_trained=True, is_roi=False).to(torch.device('cuda:%d'%7))
    conv_fea_img, fc_fea_img, out_img = model_img(img)

    roigen = ROIGenerator()
    cls_weights = list(model_img.parameters())
    weight_softmax = np.squeeze(cls_weights[-5].data.cpu().numpy())
    roi = roigen.ROIGeneration(img.cpu(), conv_fea_img, weight_softmax, label)
    model_roi = CXRClassifier(num_classes=14, is_pre_trained=True, is_roi=True).to(torch.device('cuda:%d'%7))
    var_roi = torch.autograd.Variable(roi).to(torch.device('cuda:%d'%7))
    _, fc_fea_roi, out_roi = model_roi(var_roi)

    model_fusion = FusionClassifier(input_size = 2048, output_size = 14).to(torch.device('cuda:%d'%7))
    fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
    var_fusion = torch.autograd.Variable(fc_fea_fusion).to(torch.device('cuda:%d'%7))
    out_fusion = model_fusion(var_fusion)
    print(out_fusion.size())

