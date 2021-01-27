
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

from ai_xp.config import *
#https://github.com/thtang/CheXNet-with-localization/blob/master/denseNet_localization.py
# ======= Grad CAM Function =========
class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.to(cuda)
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.cuda) if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        #self.probs = F.softmax(self.preds)[0]
        #self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        #return grads / l2_norm.data[0]
        return grads / l2_norm.item()

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data
        
        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def saveHeatmap_resize(self, filename, gcam, raw_image):
        #raw_image = self.ReadRawImage(raw_image) #resize, crop and turn to numpy
        #gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET) #L to RGB
        #gcam = cv2.addWeighted(raw_image, 0.5, gcam, 0.5, 0)
        #cv2.imwrite(filename, np.uint8(gcam))
        raw_image = self.ReadRawImage(raw_image)
        x_c, y_c = self.genHeatBoxes(gcam)

        heat_map = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET) #L to RGB
        heat_map = Image.fromarray(heat_map)#.convert('RGB')#PIL.Image
        mask_img = Image.new('RGBA', heat_map.size, color=0) #transparency
        #paste heatmap
        w, h = config['sizeX'], config['sizeY']
        upper = int(max(x_c-(w/2), 0.))
        left = int(max(y_c-(h/2), 0.))
        right = min(upper+w, heat_map.size[0])
        lower = min(left+h, heat_map.size[1])
        roi_area = (upper, left, right, lower)
        cropped_roi = heat_map.crop(roi_area)
        mask_img.paste(cropped_roi, roi_area)
        output_img = cv2.addWeighted(raw_image, 0.5, np.asarray(mask_img.convert('RGB')), 0.5, 0)

        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(output_img)
        rect = patches.Rectangle((upper, left), w, h, linewidth=2, edgecolor='b', facecolor='none')# Create a Rectangle patch
        ax.add_patch(rect)# Add the patch to the Axes
        ax.axis('off')
        fig.savefig(filename)

    def genHeatBoxes(self, data): #predicted bounding boxes
        # Find local maxima
        neighborhood_size = 100
        threshold = .1

        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        for _ in range(5):
            maxima = binary_dilation(maxima) 
        labeled, num_objects = ndimage.label(maxima)
        #slices = ndimage.find_objects(labeled)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
        """
        predBoxes = []
        for pt in xy:
            if data[int(pt[0]), int(pt[1])] > np.max(data)*.9:
                x_c = int(pt[0])
                y_c = int(pt[1])
                predBoxes.append([x_c,y_c])
        """
        x_c, y_c, data_xy = 0, 0, 0.0
        for pt in xy:
            if data[int(pt[0]), int(pt[1])] > data_xy:
                data_xy = data[int(pt[0]), int(pt[1])]
                x_c = int(pt[0])
                y_c = int(pt[1]) 

        return x_c, y_c

    def ReadRawImage(self, raw_img):
        width, height = raw_img.size   # Get dimensions
        raw_img = raw_img.resize((256, 256),Image.ANTIALIAS)
        width, height = raw_img.size   # Get dimensions
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        crop_del = (256-224)/2
        raw_img = raw_img.crop((left, top, right, bottom)) 
        raw_img= np.array(raw_img)
        return raw_img

    def saveHeatmap(self, filename, gcam, raw_image, box):
        iW, iH = raw_image.size # Get dimensions
        gcam = cv2.resize(gcam, (iW, iH))
        heat_map = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET) #L to RGB
        heat_map = Image.fromarray(heat_map)#.convert('RGB')#PIL.Image
        mask_img = Image.new('RGBA', heat_map.size, color=0) #transparency
        #paste heatmap
        upper = int(max(box[0]-(box[2]/2), 0.))
        left = int(max(box[1]-(box[3]/2), 0.))
        right = min(upper+box[2], heat_map.size[0])
        lower = min(left+box[3], heat_map.size[1])
        roi_area = (upper, left, right, lower)
        cropped_roi = heat_map.crop(roi_area)
        mask_img.paste(cropped_roi, roi_area)
        output_img = cv2.addWeighted(np.array(raw_image), 0.5, np.asarray(mask_img.convert('RGB')), 0.5, 0)

        fig, ax = plt.subplots(1)# Create figure and axes
        ax.imshow(output_img)
        rect = patches.Rectangle((upper, left), box[2], box[3], linewidth=2, edgecolor='b', facecolor='none')# Create a Rectangle patch
        ax.add_patch(rect)# Add the patch to the Axes
        ax.axis('off')
        fig.savefig(filename)

if __name__ == '__main__':
    try:
            stime = time.time()
            model_t = ATNet(num_classes=N_CLASSES, is_pre_trained=False)
            gcam = GradCAM(model=model_t, cuda=self.device) #define gradient-based class activated maps for heatmap location
            data_t = torch.unsqueeze(presImage, 0).to(self.device)
            probs = gcam.forward(data_t)
            layer_name='dense_net_121.features.denseblock4.denselayer16.conv2' 
            boxes = {}
            iW, iH = slice_s.pixel_array.shape[0], slice_s.pixel_array.shape[1] #size of original image, such as (2874,2762)
            #iW, iH = oriImage.size
            x_scale = int(iW/TRAN_SIZE)
            y_scale = int(iH/TRAN_SIZE)
            crop_del = (TRAN_SIZE-TRAN_CROP)/2
            for idx in idxs:
                gcam.backward(idx=idx)
                feat_map = gcam.generate(target_layer=layer_name)
                #gcam.saveHeatmap_resize(config['log_path']+'test_'+str(idx)+'.png', feat_map, oriImage) #save heatmap
                x_c, y_c = gcam.genHeatBoxes(feat_map)
                posX = (x_c+crop_del)*x_scale
                posY = (y_c+crop_del)*y_scale
                sizeX = config['sizeX']*x_scale
                sizeY = config['sizeY']*y_scale
                boxes[idx] = [posX, posY, sizeX, sizeY]
                #gcam.saveHeatmap(config['log_path']+'test_'+str(idx)+'.png', feat_map, oriImage, [posX, posY, sizeX, sizeY]) #for verification
            self.logger.info("Location time: {} seconds".format(time.time()-stime))
        except Exception as e:
            self.logger.info("SF_XP_L001: " + str(e))