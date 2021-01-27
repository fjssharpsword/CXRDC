# encoding: utf-8
import numpy as np
from os import listdir
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import os
import pickle
from collections import defaultdict
from collections import OrderedDict
from PIL import Image, ImageDraw
import PIL.ImageOps
import skimage
from skimage.io import *
from skimage.transform import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.measure import label as skmlabel

#self-defined
from config import *
class CAM(object):
    
    def __init__(self):
        self.TRAN_SIZE = config['TRAN_SIZE']
        self.TRAN_CROP = config['TRAN_CROP']

    # generate class activation mapping for the predicted classed
    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 224x224
        size_upsample = (self.TRAN_CROP, self.TRAN_CROP)
        bz, nc, h, w = feature_conv.shape

        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc,h*w)))
        #cam = weight_softmax[class_idx]*(feature_conv.reshape((nc,h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, size_upsample)
        return cam_img

    def compute_IoUs(self, xywh1, xywh2):
        x1, y1, w1, h1 = xywh1
        x2, y2, w2, h2 = xywh2

        dx = min(x1+w1, x2+w2) - max(x1, x2)
        dy = min(y1+h1, y2+h2) - max(y1, y2)
        intersection = dx * dy if (dx >=0 and dy >= 0) else 0.
        
        union = w1 * h1 + w2 * h2 - intersection
        IoUs = intersection / union
        
        return IoUs

    def binImage(self, heatmap):
        _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # t in the paper
        #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
        return heatmap_bin

    def selectMaxConnect(self, heatmap):
        labeled_img, num = skmlabel(heatmap, connectivity=2, background=0, return_num=True)    
        max_label = 0
        max_num = 0
        for i in range(1, num+1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        lcc = (labeled_img == max_label)
        if max_num == 0:
            lcc = (labeled_img == -1)
        lcc = lcc + 0
        return lcc 

    def returnBox2(self, data, gtbox):
        w, h = gtbox[2:]

        heatmap_bin = self.binImage(data)
        heatmap_maxconn = self.selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])

        return [minh, minw, w, h]


    def returnBox(self, data, gtbox): #predicted bounding boxes
        # Find local maxima
        neighborhood_size = 5 #10 #100
        threshold = .1
        w, h = gtbox[2:]

        data_max = filters.maximum_filter(data, neighborhood_size) #local maximum
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size) #local minimum
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        for _ in range(5):
            maxima = binary_dilation(maxima) 
        labeled, num_objects = ndimage.label(maxima)
        #slices = ndimage.find_objects(labeled)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
        #get the hot points
        """
        x_c, y_c, data_xy = 0, 0, 0.0
        for pt in xy:
            if data[int(pt[0]), int(pt[1])] > data_xy:
                data_xy = data[int(pt[0]), int(pt[1])]
                x_c = int(pt[0])
                y_c = int(pt[1]) 

        x_p = int(max(x_c-w/2, 0.))
        y_p = int(max(y_c-h/2, 0.))
        """
        x_p, y_p, iou = 0, 0, 0.0
        for pt in xy:
            x_p_t = int(max(pt[0]-w/2, 0.))
            y_p_t = int(max(pt[1]-h/2, 0.))
            iou_t = self.compute_IoUs([x_p_t, y_p_t, w, h], gtbox)
            if iou_t > iou:
                x_p = x_p_t
                y_p = y_p_t
                iou = iou_t
        
        return [x_p, y_p, w, h]

    def visHeatmap(self, batch_idx, class_name, image, cam_img, pdbox, gtbox, iou):
        #raw image 
        image = (image + 1).squeeze().permute(1, 2, 0) #[-1,1]->[1, 2]
        image = (image - image.min()) / (image.max() - image.min()) #[1, 2]->[0,1]
        image = np.uint8(255 * image) #[0,1] ->[0,255]
        
        #feature map
        heat_map = cv2.applyColorMap(np.uint8(cam_img * 255.0), cv2.COLORMAP_JET) #L to RGB
        heat_map = Image.fromarray(heat_map)#.convert('RGB')#PIL.Image
        mask_img = Image.new('RGBA', heat_map.size, color=0) #transparency
        #paste heatmap
        x1, y1, w, h = np.array(pdbox).astype(int)
        x2, y2, w, h = np.array(gtbox).astype(int)
        cropped_roi = heat_map.crop((x1,y1,x1+w,y1+h))
        mask_img.paste(cropped_roi, (x1,y1,x1+w,y1+h))
        cropped_roi = heat_map.crop((x2,y2,x2+w,y2+h))
        mask_img.paste(cropped_roi, (x2,y2,x2+w,y2+h))
        #plot
        output_img = cv2.addWeighted(image, 0.7, np.asarray(mask_img.convert('RGB')), 0.3, 0)
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(output_img)
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='b', facecolor='none')# Create a Rectangle patch
        ax.add_patch(rect)# Add the patch to the Axes
        rect = patches.Rectangle((x2, y2), w, h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
        ax.add_patch(rect)# Add the patch to the Axes
        ax.axis('off')
        fig.savefig(config['img_path']+ str(batch_idx+1) +'_'+class_name+'_'+ str(iou)[0:6] +'.png')
        """
        image = Image.fromarray(image).convert('RGB')#PIL.Image
        x2, y2, w, h = np.array(gtbox).astype(int)
        cropped_roi = image.crop((x2,y2,x2+w,y2+h))
        width, height = image.size
        cropped_roi = cropped_roi.resize((width, height),Image.ANTIALIAS)
        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(cropped_roi)
        ax.axis('off')
        fig.savefig(config['img_path']+str(batch_idx+1)+'_'+class_name+'.png')
        """




    