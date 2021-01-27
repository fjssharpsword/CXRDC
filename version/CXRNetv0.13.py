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
from config import *
from net.AENet import AENet
from net.UNet import UNet

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(ImageClassifier, self).__init__()

        self.model = AENet(num_classes=N_CLASSES, is_pre_trained=True)#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_aenet.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            self.model.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained AENet model checkpoint: "+CKPT_PATH)
        else:
            print('No required model')

    def forward(self, x):
        #x: N*C*W*H
        feat, out, _ = self.model(x)
        return feat, out

class RegionClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(RegionClassifier, self).__init__()
        #segmentation module
        self.model_unet = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            self.model_unet.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        else:
            print('No required model')
        #classification 
        self.msa = MultiScaleAttention()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        
    def forward(self, x):
        #x: N*C*W*H
        out_mask = self.model_unet(x)
        out_mask = out_mask.ge(0.5).float() #0,1 binarization
        out_mask = out_mask.repeat(1, 3, 1, 1) #torch.stack([out_mask, out_mask, out_mask], 1).squeeze()
        x = torch.mul(x, out_mask)
        """#visualizetion
        image = x[0,:,:,:].squeeze().permute(1, 2, 0).cpu().numpy()
        image = np.uint8(255 * image) #[0,1] ->[0,255]
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(image)
        ax.axis('off')
        fig.savefig(config['img_path'] + 'mask.png')
        """

        #region classifier
        x = self.msa(x) * x
        x = self.dense_net_121.features(x)
        x = F.relu(x, inplace=True)
        feat = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)
        out = self.classifier(feat)

        return feat, out

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

class FusionClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(FusionClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, fusion_var):
        out = self.fc(fusion_var)
        out = self.Sigmoid(out)

        return out

if __name__ == "__main__":
    #for debug   
    img = torch.rand(32, 3, 224, 224)#.to(torch.device('cuda:%d'%4))
    var_img = torch.autograd.Variable(img).to(torch.device('cuda:%d'%4))

    model_img = ImageClassifier(num_classes=14, is_pre_trained=True).to(torch.device('cuda:%d'%4))
    fc_fea_img, out_img = model_img(var_img)

    model_roi = RegionClassifier(num_classes=14, is_pre_trained=True).to(torch.device('cuda:%d'%4))
    fc_fea_roi, out_roi, out_x = model_roi(var_img)

    model_fusion = FusionClassifier(input_size = 2048, output_size = 14).to(torch.device('cuda:%d'%4))
    fc_fea_fusion = torch.cat((fc_fea_img,fc_fea_roi), 1)
    var_fusion = torch.autograd.Variable(fc_fea_fusion).to(torch.device('cuda:%d'%4))
    out_fusion = model_fusion(var_fusion)
    print(out_fusion.size())

