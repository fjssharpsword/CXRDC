# encoding: utf-8
"""
Unifying Self-supervised and Pathological Region Learning
Author: Jason.Fang
Update time: 19/01/2021
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
#from net.UNet import UNet

class CXRNet(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(CXRNet, self).__init__()

        self.msa = MultiScaleAttention()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        
    def forward(self, x):
        #x: N*C*W*H
        x = self.msa(x) * x
        x = self.dense_net_121.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

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

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = CXRNet(num_classes=14, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    out  = model(x)
    print(out.size())

