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
      
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        #self.adpool = nn.AdaptiveAvgPool2d((7, 7))
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        
    def forward(self, x, mask):
        #x: N*C*W*H
        x = self.dense_net_121.features(x)
        """
        #show
        cam = x[0].sum(0).detach().cpu().squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.applyColorMap(np.uint8(cam * 255.0), cv2.COLORMAP_JET) #L to RGB
        #cam = Image.fromarray(cam)#.convert('RGB')#PIL.Image
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(cam)
        ax.axis('off')
        fig.savefig('/data/pycode/CXR-IRNet/imgs/cam_before.png')
        """
    
        mask = self.adpool(mask) #bz*1*224*224->bz*1*7*7
        mask = mask.ge(0.5).float() #0,1 binarization
        mask = mask.repeat(1, x.size(1), 1, 1) #bz*1*7*7 -> bz*1024*7*7
        x = torch.mul(x, mask)
        
        """
        #show
        cam = x[0].sum(0).detach().cpu().squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.applyColorMap(np.uint8(cam * 255.0), cv2.COLORMAP_JET) #L to RGB
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(cam)
        ax.axis('off')
        fig.savefig('/data/pycode/CXR-IRNet/imgs/cam_after.png')
        """
        #out = self.classifier(torch.mul(out_global, out_local)) #1024
        #out = self.classifier(torch.cat((out_global, out_local), 1)) #2048
        x = self.pool(x).view(x.size(0), -1)
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = CXRNet(num_classes=14, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    out  = model(x)
    print(out.size())

