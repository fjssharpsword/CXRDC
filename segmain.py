# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Email: 11949039@mail.sustech.edu.cn
Update time: 20/01/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
#self-defined
from dataset.jsrt import get_train_dataloader
from util.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from net.UNet import UNet, DiceLoss
from util.logger import get_logger
from config import *

#command parameters
parser = argparse.ArgumentParser(description='For Pathological Region Segmentation')
parser.add_argument('--model', type=str, default='UNet', help='UNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def SegmentationTrain():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    model = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
    CKPT_PATH = config['CKPT_PATH'] +  'best_unet_heart.pkl'
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)

    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size = 10, gamma = 1)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.train()  #set model to training mode
    #ce_criterion = nn.CrossEntropyLoss() #define cross-entropy loss
    dice_criterion = DiceLoss()
    print('********************load data succeed!********************')

    print('********************begin training!********************')
    dice_loss = 1.0
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, mask_left, mask_right, mask_heart) in enumerate(dataloader_train):
                optimizer_model.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask_heart).cuda()
                out_mask = model(var_image)
                #backward    
                mask_loss = dice_criterion(out_mask, var_mask)
                mask_loss.backward()
                optimizer_model.step()
                train_loss.append(mask_loss.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%mask_loss.item()) ))
                sys.stdout.flush()        
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss)))

        #save checkpoint
        if dice_loss > np.mean(train_loss):
            dice_loss = np.mean(train_loss)
            #torch.save(model.module.state_dict(), CKPT_PATH)
            torch.save(model.state_dict(), config['CKPT_PATH'] +  'best_unet_heart.pkl') #checkpoint
            print(' Epoch: {} model has been already save!'.format(epoch+1))
    
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def main():
    SegmentationTrain()

if __name__ == '__main__':
    main()