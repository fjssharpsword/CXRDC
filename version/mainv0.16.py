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
        #for left_lung
        model_unet_left = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_left.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet_left.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_left.eval()

        model_left = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        optimizer_left = optim.Adam(model_left.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_left = lr_scheduler.StepLR(optimizer_left, step_size = 10, gamma = 1)
        #for right lung
        model_unet_right = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_right.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet_right.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_right.eval()

        model_right = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        optimizer_right = optim.Adam(model_right.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_right = lr_scheduler.StepLR(optimizer_right, step_size = 10, gamma = 1)
        #for heart
        model_unet_heart = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_heart.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet_heart.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_heart.eval()

        model_heart = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        optimizer_heart = optim.Adam(model_heart.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_heart = lr_scheduler.StepLR(optimizer_heart, step_size = 10, gamma = 1)
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
        model_left.train()  #set model to training mode
        model_right.train()
        model_heart.train()
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer_left.zero_grad()
                optimizer_right.zero_grad() 
                optimizer_heart.zero_grad() 
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                #for left lung
                mask = model_unet_left(var_image)
                roi = ROIGeneration(image, mask)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_left = model_left(var_roi)#forward
                loss_left = bce_criterion(out_left, var_label)
                loss_left.backward()
                optimizer_left.step()
                #for right lung
                mask = model_unet_right(var_image)
                roi = ROIGeneration(image, mask)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_right = model_right(var_roi)#forward
                loss_right = bce_criterion(out_right, var_label)
                loss_right.backward()
                optimizer_right.step()
                #for heart
                mask = model_unet_heart(var_image)
                roi = ROIGeneration(image, mask)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_heart = model_heart(var_roi)#forward
                loss_heart = bce_criterion(out_heart, var_label)
                loss_heart.backward()
                optimizer_heart.step()
                #loss sum 
                loss_tensor = loss_left + loss_right + loss_heart
                train_loss.append(loss_tensor.item())
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()        
        lr_scheduler_left.step()  #about lr and gamma
        lr_scheduler_right.step()
        lr_scheduler_heart.step()
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model_left.eval() #turn to test mode
        model_right.eval()
        model_heart.eval()
        val_loss = []
        gt = torch.FloatTensor().cuda()
        preds = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                pred = torch.FloatTensor().cuda()
                gt = torch.cat((gt, label.cuda()), 0)
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                #for left lung
                mask = model_unet_left(var_image)
                roi = ROIGeneration(image, mask)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_left = model_left(var_roi)#forward
                pred = torch.cat((pred, out_left.data.unsqueeze(0)), 0)
                #for right lung
                mask = model_unet_right(var_image)
                roi = ROIGeneration(image, mask)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_right = model_right(var_roi)#forward
                pred = torch.cat((pred, out_right.data.unsqueeze(0)), 0)
                #for heart
                mask = model_unet_heart(var_image)
                roi = ROIGeneration(image, mask)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_heart = model_heart(var_roi)#forward
                pred = torch.cat((pred, out_heart.data.unsqueeze(0)), 0)
                #prediction
                pred = torch.max(pred, 0)[0] #torch.mean
                preds = torch.cat((preds, pred.data), 0)
                loss_tensor = bce_criterion(pred, var_label)
                val_loss.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        #evaluation       
        AUROCs_avg = np.array(compute_AUCs(gt, preds)).mean()
        print("\r Eopch: %5d validation loss = %.6f, average AUROC=%.4f"% (epoch + 1, np.mean(val_loss), AUROCs_avg)) 

        #save checkpoint
        if AUROC_best < AUROCs_avg:
            AUROC_best = AUROCs_avg
            torch.save(model_img.state_dict(), config['CKPT_PATH'] +  'left_model.pkl') #Saving torch.nn.DataParallel Models
            torch.save(model_roi.state_dict(), config['CKPT_PATH'] + 'right_model.pkl')
            torch.save(model_fusion.state_dict(), config['CKPT_PATH'] + 'heart_model.pkl')
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
        #for left
        model_left = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        CKPT_PATH = config['CKPT_PATH']  +'left_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_left.load_state_dict(checkpoint) #strict=False
        print("=> loaded left model checkpoint: "+CKPT_PATH)
        model_left.eval()

        model_unet_left = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_left.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_unet_left.load_state_dict(checkpoint) #strict=False
        print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_left.eval()

        #for right
        model_right = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        CKPT_PATH = config['CKPT_PATH']  +'right_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_right.load_state_dict(checkpoint) #strict=False
        print("=> loaded right model checkpoint: "+CKPT_PATH)
        model_right.eval()

        model_unet_right = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet_right.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_unet_right.load_state_dict(checkpoint) #strict=False
        print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet_right.eval()

        #for heart
        model_heart = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        CKPT_PATH = config['CKPT_PATH']  +'heart_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_heart.load_state_dict(checkpoint) #strict=False
        print("=> loaded heart model checkpoint: "+CKPT_PATH)
        model_heart.eval()

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
    preds = torch.FloatTensor().cuda()
    # switch to evaluate mode
    model_img.eval() #turn to test mode
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            gt = torch.cat((gt, label.cuda()), 0)
            pred = torch.FloatTensor().cuda()
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            #for left lung
            mask = model_unet_left(var_image)
            roi = ROIGeneration(image, mask)
            var_roi = torch.autograd.Variable(roi).cuda()
            out_left = model_left(var_roi)#forward
            pred = torch.cat((pred, out_left.data.unsqueeze(0)), 0)
            #for right lung
            mask = model_unet_right(var_image)
            roi = ROIGeneration(image, mask)
            var_roi = torch.autograd.Variable(roi).cuda()
            out_right = model_right(var_roi)#forward
            pred = torch.cat((pred, out_right.data.unsqueeze(0)), 0)
            #for heart
            mask = model_unet_heart(var_image)
            roi = ROIGeneration(image, mask)
            var_roi = torch.autograd.Variable(roi).cuda()
            out_heart = model_heart(var_roi)#forward
            pred = torch.cat((pred, out_heart.data.unsqueeze(0)), 0)
            #prediction
            pred = torch.max(pred, 0)[0] #torch.mean
            preds = torch.cat((preds, pred.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #for evaluation
    AUROC_img = compute_AUCs(gt, preds)
    AUROC_avg = np.array(AUROC_img).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_img[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()