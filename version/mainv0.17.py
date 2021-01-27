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
#from dataset.CXR8Benchmark import get_test_dataloader, get_train_val_dataloader
#from dataset.jsrt import get_train_dataloader as get_train_dataloader
from dataset.CXR8Common import get_train_dataloader, get_validation_dataloader, get_test_dataloader, get_bbox_dataloader
from util.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from util.CAM import CAM
from net.UNet import UNet, DiceLoss
from net.CXRNet import CXRClassifier, ROIGeneration
from util.logger import get_logger
from config import *

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='CXRNet', help='CXRNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    #dataloader_train, dataloader_val = get_train_val_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8, split_ratio=0.1)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        
        model = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size = 10, gamma = 1)

        #for left_lung
        model_unet_left = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_left.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet_left.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_left.eval()
        #for right lung
        model_unet_right = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_right.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet_right.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_right.eval()
        #for heart
        model_unet_heart = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_heart.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet_heart.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_heart.eval()
    else: 
        print('No required model')
        return #over

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    bce_criterion = nn.BCELoss() #define binary cross-entropy loss
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer_model.zero_grad()
                #pathological regions generation
                var_image = torch.autograd.Variable(image).cuda()         
                mask_left = model_unet_left(var_image)#for left lung
                mask_right = model_unet_right(var_image)#for right lung
                mask_heart = model_unet_heart(var_image)#for heart 
                patchs, patch_labels, globals, global_labels = ROIGeneration(image, [mask_left, mask_right, mask_heart], label) 
                #training
                loss_patch, loss_global = torch.FloatTensor([0.0]).cuda(), torch.FloatTensor([0.0]).cuda()
                if len(patchs)>0:
                    var_patchs = torch.autograd.Variable(patchs).cuda()
                    var_patch_labels = torch.autograd.Variable(patch_labels).cuda()
                    out_patch = model(var_patchs, is_patch = True)#forward
                    loss_patch = bce_criterion(out_patch, var_patch_labels)
                """
                if len(globals)>0:
                    var_globals = torch.autograd.Variable(globals).cuda()
                    var_global_labels = torch.autograd.Variable(global_labels).cuda()
                    out_global = model(var_globals, is_patch = False)#forward
                    loss_global = bce_criterion(out_global, var_global_labels)
                """
                loss_tensor = loss_patch + loss_global
                loss_tensor.backward()
                optimizer_model.step()
                train_loss.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model.eval() #turn to test mode
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                #pathological regions generation
                var_image = torch.autograd.Variable(image).cuda()         
                mask_left = model_unet_left(var_image)#for left lung
                mask_right = model_unet_right(var_image)#for right lung
                mask_heart = model_unet_heart(var_image)#for heart 
                patchs, patch_labels, globals, global_labels = ROIGeneration(image, [mask_left, mask_right, mask_heart], label) 
                #training
                loss_patch, loss_global = torch.FloatTensor([0.0]).cuda(), torch.FloatTensor([0.0]).cuda()
                if len(patchs)>0:
                    var_patchs = torch.autograd.Variable(patchs).cuda()
                    var_patch_labels = torch.autograd.Variable(patch_labels).cuda()
                    out_patch = model(var_patchs, is_patch = True)#forward
                    loss_patch = bce_criterion(out_patch, var_patch_labels)
                    gt = torch.cat((gt, patch_labels.cuda()), 0)
                    pred = torch.cat((pred, out_patch.data), 0)
                """
                if len(globals)>0:
                    var_globals = torch.autograd.Variable(globals).cuda()
                    var_global_labels = torch.autograd.Variable(global_labels).cuda()
                    out_global = model(var_globals, is_patch = False)#forward
                    loss_global = bce_criterion(out_global, var_global_labels)
                    gt = torch.cat((gt, global_labels.cuda()), 0)
                    pred = torch.cat((pred, out_global.data), 0)
                """
                loss_tensor = loss_patch + loss_global
                val_loss.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        #evaluation       
        AUROCs_avg = np.array(compute_AUCs(gt, pred)).mean()
        print("\r Eopch: %5d validation loss = %.6f, average AUROC=%.4f"% (epoch + 1, np.mean(val_loss), AUROCs_avg)) 

        #save checkpoint
        if AUROC_best < AUROCs_avg:
            AUROC_best = AUROCs_avg
            torch.save(model.state_dict(), config['CKPT_PATH'] +  'best_model_CXRNet.pkl') 
            print(' Epoch: {} model has been already save!'.format(epoch+1))
    
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        CKPT_PATH = config['CKPT_PATH']  +'best_model_CXRNet.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded left model checkpoint: "+CKPT_PATH)
        model.eval()
        #for left lung
        model_unet_left = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_left.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_unet_left.load_state_dict(checkpoint) #strict=False
        print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_left.eval()
        #for right lung
        model_unet_right = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_right.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_unet_right.load_state_dict(checkpoint) #strict=False
        print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_right.eval()
        #for heart
        model_unet_heart = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_right.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_unet_heart.load_state_dict(checkpoint) #strict=False
        print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_heart.eval()
        
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
        
    print('******* begin testing!*********')
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            #pathological regions generation
            var_image = torch.autograd.Variable(image).cuda()         
            mask_left = model_unet_left(var_image)#for left lung
            mask_right = model_unet_right(var_image)#for right lung
            mask_heart = model_unet_heart(var_image)#for heart 
            patchs, patch_labels, globals, global_labels = ROIGeneration(image, [mask_left, mask_right, mask_heart], label) 
            #training
            if len(patchs)>0:
                var_patchs = torch.autograd.Variable(patchs).cuda()
                var_patch_labels = torch.autograd.Variable(patch_labels).cuda()
                out_patch = model(var_patchs, is_patch = True)#forward
                gt = torch.cat((gt, patch_labels.cuda()), 0)
                pred = torch.cat((pred, out_patch.data), 0)
            if len(globals)>0:
                var_globals = torch.autograd.Variable(globals).cuda()
                var_global_labels = torch.autograd.Variable(global_labels).cuda()
                out_global = model(var_globals, is_patch = False)#forward
                gt = torch.cat((gt, global_labels.cuda()), 0)
                pred = torch.cat((pred, out_global.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #for evaluation
    AUROC_img = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROC_img).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_img[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()