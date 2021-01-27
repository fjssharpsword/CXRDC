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

#define myself
from config import *
#construct model
class CXRClassifier(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True, is_roi=False):
        super(CXRClassifier, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        self.msa = MultiScaleAttention()
        self.is_roi =  is_roi
        
    def forward(self, x):
        #x: N*C*W*H
        """
        x = self.msa(x) * x
        x = self.dense_net_121(x)
        return x
        """
        if self.is_roi == False: #for image training
            x = self.msa(x) * x
            conv_fea = self.dense_net_121.features(x)
            out = F.relu(conv_fea, inplace=True)
            fc_fea = F.avg_pool2d(out, kernel_size=7, stride=1).view(conv_fea.size(0), -1)
            out = self.dense_net_121.classifier(fc_fea)
            return conv_fea, fc_fea, out
        else: #roi model
            out = self.dense_net_121.classifier(x)
            return out
        
        
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

class ROIGenerator(object):
    def __init__(self):
        super(ROIGenerator, self).__init__()
        self.transform_seq = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                ])
        self.regions = self.get_regions(7,7,level_n=3)

    def ROIGeneration(self, fm_cuda):
        # feature map -> feature mask (using feature map to crop on the original image) -> crop -> patchs
        feature_conv = fm_cuda.cpu()

        final_fea = None
        for _, r in enumerate(self.regions):
            st_x, st_y, ed_x, ed_y = r
            region_fea = (feature_conv[:, :, st_x: ed_x, st_y: ed_y].max(dim=3)[0]).max(dim=2)[0]#max-pooling
            region_fea = region_fea / torch.norm(region_fea, dim=1, keepdim=True)#normalization
            if final_fea is None:
                final_fea = region_fea
            else:
                final_fea = final_fea + region_fea
           
        return final_fea #batch_size X 1024
    
    def get_regions(self, h: int, w: int, level_n=3):
        """
        Divide the image into several regions.
        Args:
            h (int): height for dividing regions.
            w (int): width for dividing regions.
        Returns:
            regions (List): a list of region positions.
        """
        m = 1
        n_h, n_w = 1, 1
        regions = list()
        if h != w:
            min_edge = min(h, w)
            left_space = max(h, w) - min(h, w)
            iou_target = 0.4
            iou_best = 1.0
            while True:
                iou_tmp = (min_edge ** 2 - min_edge * (left_space // m)) / (min_edge ** 2)
                # small m maybe result in non-overlap
                if iou_tmp <= 0:
                    m += 1
                    continue

                if abs(iou_tmp - iou_target) <= iou_best:
                    iou_best = abs(iou_tmp - iou_target)
                    m += 1
                else:
                    break
            if h < w:
                n_w = m
            else:
                n_h = m

        for i in range(level_n):
            region_width = int(2 * 1.0 / (i + 2) * min(h, w))
            step_size_h = (h - region_width) // n_h
            step_size_w = (w - region_width) // n_w

            for x in range(n_h):
                for y in range(n_w):
                    st_x = step_size_h * x
                    ed_x = st_x + region_width - 1
                    assert ed_x < h
                    st_y = step_size_w * y
                    ed_y = st_y + region_width - 1
                    assert ed_y < w
                    regions.append((st_x, st_y, ed_x, ed_y))

            n_h += 1
            n_w += 1

        return regions

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

