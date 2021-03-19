# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Email: 11949039@mail.sustech.edu.cn
Update time: 28/12/2020
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
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
#define by myself
from config import *
from net.CXRNet import ImageClassifier
from net.UNet import UNet
from util.logger import get_logger
from dataset.NIHCXR import get_train_dataloader, get_test_dataloader, get_bbox_dataloader
from util.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from util.CAM import CAM

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
    dataloader_val = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model_img = ImageClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] + 'best_img_model.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_img.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained image model checkpoint: "+CKPT_PATH)
        optimizer_img = optim.Adam(model_img.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_img = lr_scheduler.StepLR(optimizer_img, step_size = 10, gamma = 1)

        # initialize and load the model
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

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    bce_criterion = nn.BCELoss() #define binary cross-entropy loss
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model_img.train()  #set model to training mode
        train_img_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer_img.zero_grad()
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                #image-level
                var_mask = model_unet(var_image)
                out_img = model_img(var_image, var_mask)#forward
                loss_img = bce_criterion(out_img, var_label)
                loss_img.backward()
                optimizer_img.step()
                train_img_loss.append(loss_img.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train image loss ={}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_img.item()) ))
                sys.stdout.flush()    
        lr_scheduler_img.step()  #about lr and gamma
        print("\r Eopch: %5d train image loss = %.6f" % (epoch + 1, np.mean(train_img_loss ))) 

        model_img.eval() #turn to test mode
        test_img_loss = []
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                gt = torch.cat((gt, label.cuda()), 0)
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_mask = model_unet(var_image)
                out_img = model_img(var_image, var_mask)#forward
                loss_img = bce_criterion(out_img, var_label)
                test_img_loss.append(loss_img.item())
                pred = torch.cat((pred, out_img.data), 0)
                sys.stdout.write('\r Epoch: {} / Step: {} : test image loss ={}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_img.item()) ))
                sys.stdout.flush()
        #evaluation
        AUROCs_img = np.array(compute_AUCs(gt, pred)).mean()
        logger.info("\r Eopch: %5d test loss = %.6f and AUROC = %.4f" % (epoch + 1, np.mean(test_img_loss), AUROCs_img)) 
        #save checkpoint
        if AUROC_best < AUROCs_img:
            AUROC_best = AUROCs_img
            torch.save(model_img.state_dict(), config['CKPT_PATH'] +  'best_img_model.pkl') 
            print(' Epoch: {} model has been already save!'.format(epoch+1))
    
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def BoxTest():
    print('********************load data********************')
    dataloader_bbox = get_bbox_dataloader(batch_size=1, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model_img = ImageClassifier(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        CKPT_PATH = config['CKPT_PATH'] + 'best_img_model.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model_img.load_state_dict(checkpoint) #strict=False
            print("=> loaded well-trained image model checkpoint: "+CKPT_PATH)
        model_img.eval()

        # initialize and load the model
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
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin bounding box testing!*********')
    #np.set_printoptions(suppress=True) #to float
    #for name, layer in model.named_modules():
    #    print(name, layer)
    cls_weights = list(model_img.parameters())
    weight_softmax = np.squeeze(cls_weights[-5].data.cpu().numpy())
    cam = CAM()
    IoUs = []
    IoU_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    with torch.autograd.no_grad():
        for batch_idx, (_, gtbox, image, label) in enumerate(dataloader_bbox):
            var_image = torch.autograd.Variable(image).cuda()
            out_img = model_img(var_image)
            idx = torch.where(label[0]==1)[0] #true label
            cam_img = cam.returnCAM(conv_fea_img.cpu().data.numpy(), weight_softmax, idx)
            pdbox = cam.returnBox(cam_img, gtbox[0].numpy())
            iou = compute_IoUs(pdbox, gtbox[0].numpy())
            IoU_dict[idx.item()].append(iou)
            IoUs.append(iou) #compute IoU
            if iou>0.99: 
                cam.visHeatmap(batch_idx, CLASS_NAMES[idx], image, cam_img, pdbox, gtbox[0].numpy(), iou) #visulization
            sys.stdout.write('\r box process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    print('The average IoU is {:.4f}'.format(np.array(IoUs).mean()))
    for i in range(len(IoU_dict)):
        print('The average IoU of {} is {:.4f}'.format(CLASS_NAMES[i], np.array(IoU_dict[i]).mean())) 

def main():
    Train() #for training
    #BoxTest() #for test
if __name__ == '__main__':
    main()