# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Email: 11949039@mail.sustech.edu.cn
Update time: 19/01/2021
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
from dataset.CXR8Common import get_train_dataloader, get_validation_dataloader, get_test_dataloader, get_bbox_dataloader
#from dataset.CXR8Benchmark import get_test_dataloader, get_train_val_dataloader
#from dataset.jsrt import get_train_dataloader as get_train_dataloader
from util.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs, seg_visulization
from util.CAM import CAM
from net.CXRNet import CXRNet
from net.UNet import UNet, DiceLoss
from util.logger import get_logger
from config import *

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='CXRNet', help='CXRNet')
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
    CKPT_PATH = config['CKPT_PATH'] +  'best_unet.pkl'
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
            for batch_idx, (image, mask) in enumerate(dataloader_train):
                optimizer_model.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
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
            torch.save(model.state_dict(), config['CKPT_PATH'] +  'best_unet.pkl') #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))
    
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    #dataloader_train, dataloader_val = get_train_val_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size = 10, gamma = 1)
        torch.backends.cudnn.benchmark = True  # improve train speed slightly
        bce_criterion = nn.BCELoss() #define binary cross-entropy loss
        #mse_criterion = nn.MSELoss() #define regression loss

        model_unet = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet.eval()
    else: 
        print('No required model')
        return #over
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        train_loss = []
        model.train()  #set model to training mode
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer_model.zero_grad()
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()

                var_mask = model_unet(var_image)
                var_mask = var_mask.ge(0.5).float() #0,1 binarization
                mask_np = var_mask.squeeze().cpu().numpy() #bz*224*224
                patchs = torch.FloatTensor()
                for i in range(0, mask_np.shape[0]):
                    mask = mask_np[i]
                    ind = np.argwhere(mask != 0)
                    if len(ind)>2:
                        minh = min(ind[:,0])
                        minw = min(ind[:,1])
                        maxh = max(ind[:,0])
                        maxw = max(ind[:,1])

                        image_crop = image[i].permute(1,2,0).squeeze().numpy() #224*224*3
                        image_crop = image_crop[minh:maxh,minw:maxw,:]
                        image_crop = cv2.resize(image_crop, (config['TRAN_CROP'],config['TRAN_CROP']))
                        image_crop = torch.FloatTensor(image_crop).permute(2, 1, 0).unsqueeze(0) #1*3*224*224
                        patchs = torch.cat((patchs, image_crop), 0)
                    else:
                        image_crop = image[i].unsqueeze(0)
                        patchs = torch.cat((patchs, image_crop), 0)

                var_patchs = torch.autograd.Variable(patchs).cuda()
                var_output = model(var_patchs)#forward
                loss_tensor = bce_criterion(var_output, var_label)
                loss_tensor.backward()
                optimizer_model.step()
                train_loss.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train BCE loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()        
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model.eval() #turn to test mode
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                gt = torch.cat((gt, label.cuda()), 0)
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()

                var_mask = model_unet(var_image)
                var_mask = var_mask.ge(0.5).float() #0,1 binarization
                mask_np = var_mask.squeeze().cpu().numpy() #bz*224*224
                patchs = torch.FloatTensor()
                for i in range(0, mask_np.shape[0]):
                    mask = mask_np[i]
                    ind = np.argwhere(mask != 0)
                    if len(ind)>0:
                        minh = min(ind[:,0])
                        minw = min(ind[:,1])
                        maxh = max(ind[:,0])
                        maxw = max(ind[:,1])

                        image_crop = image[i].permute(1,2,0).squeeze().numpy() #224*224*3
                        image_crop = image_crop[minh:maxh,minw:maxw,:]
                        image_crop = cv2.resize(image_crop, (config['TRAN_CROP'],config['TRAN_CROP']))
                        image_crop = torch.FloatTensor(image_crop).permute(2, 1, 0).unsqueeze(0) #1*3*224*224
                        patchs = torch.cat((patchs, image_crop), 0)
                    else:
                        image_crop = image[i].unsqueeze(0)
                        patchs = torch.cat((patchs, image_crop), 0)

                var_patchs = torch.autograd.Variable(patchs).cuda()
                var_output = model(var_patchs)#forward
                loss_tensor = bce_criterion(var_output, var_label)
                pred = torch.cat((pred, var_output.data), 0)
                val_loss.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        #evaluation       
        AUROCs_avg = np.array(compute_AUCs(gt, pred)).mean()
        logger.info("\r Eopch: %5d validation loss = %.6f, Validataion AUROC image=%.4f" % (epoch + 1, np.mean(val_loss), AUROCs_avg)) 

        #save checkpoint
        if AUROC_best < AUROCs_avg:
            AUROC_best = AUROCs_avg
            torch.save(model.state_dict(), config['CKPT_PATH'] +  'best_model_CXRNet.pkl') #Saving torch.nn.DataParallel Models
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
        model = CXRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()
        CKPT_PATH = config['CKPT_PATH']  +'best_model_CXRNet.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded Image model checkpoint: "+CKPT_PATH)
        torch.backends.cudnn.benchmark = True  # improve train speed slightly

        model_unet = UNet(n_channels=3, n_classes=1).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] +  'best_unet.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_unet.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained unet model checkpoint: "+CKPT_PATH)
        model_unet.eval()
    else: 
        print('No required model')
        return #over
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            gt = torch.cat((gt, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()

            var_mask = model_unet(var_image)
            var_mask = var_mask.ge(0.5).float() #0,1 binarization
            mask_np = var_mask.squeeze().cpu().numpy() #bz*224*224
            patchs = torch.FloatTensor()
            for i in range(0, mask_np.shape[0]):
                mask = mask_np[i]
                ind = np.argwhere(mask != 0)
                if len(ind)>0:
                    minh = min(ind[:,0])
                    minw = min(ind[:,1])
                    maxh = max(ind[:,0])
                    maxw = max(ind[:,1])

                    image_crop = image[i].permute(1,2,0).squeeze().numpy() #224*224*3
                    image_crop = image_crop[minh:maxh,minw:maxw,:]
                    image_crop = cv2.resize(image_crop, (config['TRAN_CROP'],config['TRAN_CROP']))
                    image_crop = torch.FloatTensor(image_crop).permute(2, 1, 0).unsqueeze(0) #1*3*224*224
                    patchs = torch.cat((patchs, image_crop), 0)
                else:
                    image_crop = image[i].unsqueeze(0)
                    patchs = torch.cat((patchs, image_crop), 0)

            var_patchs = torch.autograd.Variable(patchs).cuda()
            var_output = model(var_patchs)#forward
            pred = torch.cat((pred, var_output.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #for evaluation
    AUROC_all = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROC_all).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_all[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

def main():
    #SegmentationTrain()
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()
            