# encoding: utf-8
"""
AutoEncoder for self-supervised learning
Author: Jason.Fang
Update time: 18/01/2021
"""
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

class AENet(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(AENet, self).__init__()

        ## encoder layers ##
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(64, 3, 2, stride=2)

        #common layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gem = GeneralizedMeanPooling()
        self.classifer = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())

    def forward(self, x):
        #x: N*C*W*H
        x = self.dense_net_121.features(x)
        feat = self.gem(x) 
        feat = feat.view(feat.size(0),-1)
        out = self.classifer(feat)

        # add transpose conv layers, with relu activation function
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        x = self.relu(self.t_conv3(x))
        x = self.relu(self.t_conv4(x))
        x = self.relu(self.t_conv5(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = self.sigmoid(x)

        return feat, out, x

#https://github.com/naver/deep-image-retrieval/blob/master/dirtorch/nets/layers/pooling.py
#https://arxiv.org/pdf/1711.02512.pdf
class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        #self.p = float(norm)
        self.p = nn.Parameter(torch.ones(1) * norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'
            
if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = AENet(num_classes=14, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    feat, out, x = model(x)
    print(feat.size())
    print(out.size())
    print(x.size())