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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
#self-defined
#from dataset.CXR8Benchmark import get_test_dataloader, get_train_val_dataloader
from dataset.CXR8Common import get_train_dataloader, get_validation_dataloader, get_test_dataloader, get_bbox_dataloader
from util.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from util.CAM import CAM
from net.CXRNet import CXRClassifier, ROIGenerator, FusionClassifier
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
        model_img = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True, is_roi=False).cuda()#initialize model 
        #model_img = nn.DataParallel(model_img).cuda()  # make model available multi GPU cores training
        optimizer_img = optim.Adam(model_img.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_img = lr_scheduler.StepLR(optimizer_img, step_size = 10, gamma = 1)

        roigen = ROIGenerator()

        model_roi = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True, is_roi=True).cuda()
        #model_roi = nn.DataParallel(model_roi).cuda()
        optimizer_roi = optim.Adam(model_roi.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_roi = lr_scheduler.StepLR(optimizer_roi, step_size = 10, gamma = 1)

        model_fusion = FusionClassifier(input_size=2048, output_size=N_CLASSES).cuda()
        #model_fusion = nn.DataParallel(model_fusion).cuda()
        optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion, step_size = 10, gamma = 1)
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
        model_roi.train()
        model_fusion.train()
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer_img.zero_grad()
                optimizer_roi.zero_grad() 
                optimizer_fusion.zero_grad() 
                #image-level
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                conv_fea_img, fc_fea_img, out_img = model_img(var_image)#forward
                loss_img = bce_criterion(out_img, var_label)
                loss_img.backward()
                optimizer_img.step()
                #ROI-level
                roi = roigen.ROIGeneration(conv_fea_img)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_roi = model_roi(var_roi)
                loss_roi = bce_criterion(out_roi, var_label) 
                loss_roi.backward()
                optimizer_roi.step()
                #Fusion
                fc_fea_fusion = torch.cat((fc_fea_img,var_roi), 1)
                var_fusion = torch.autograd.Variable(fc_fea_fusion).cuda()
                out_fusion = model_fusion(var_fusion)
                loss_fusion = bce_criterion(out_fusion, var_label) 
                loss_fusion.backward()
                optimizer_fusion.step() 
                #loss sum 
                loss_tensor = loss_img + loss_roi + loss_fusion 
                train_loss.append(loss_tensor.item())
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : image loss ={}, roi loss ={}, fusion loss = {}, train loss = {}'
                                .format(epoch+1, batch_idx+1, float('%0.6f'%loss_img.item()), float('%0.6f'%loss_roi.item()),
                                float('%0.6f'%loss_fusion.item()), float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()        
        lr_scheduler_img.step()  #about lr and gamma
        lr_scheduler_roi.step()
        lr_scheduler_fusion.step()
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model_img.eval() #turn to test mode
        model_roi.eval()
        model_fusion.eval()
        loss_img_all, loss_roi_all, loss_fusion_all = [], [], []
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred_img = torch.FloatTensor().cuda()
        pred_roi = torch.FloatTensor().cuda()
        pred_fusion = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                gt = torch.cat((gt, label.cuda()), 0)
                #image-level
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                conv_fea_img, fc_fea_img, out_img = model_img(var_image)#forward
                loss_img = bce_criterion(out_img, var_label) 
                pred_img = torch.cat((pred_img, out_img.data), 0)
                #ROI-level
                roi = roigen.ROIGeneration(conv_fea_img)
                var_roi = torch.autograd.Variable(roi).cuda()
                out_roi = model_roi(var_roi)
                loss_roi = bce_criterion(out_roi, var_label) 
                pred_roi = torch.cat((pred_roi, out_roi.data), 0)
                #Fusion
                fc_fea_fusion = torch.cat((fc_fea_img,var_roi), 1)
                var_fusion = torch.autograd.Variable(fc_fea_fusion).cuda()
                out_fusion = model_fusion(var_fusion)
                loss_fusion = bce_criterion(out_fusion, var_label) 
                pred_fusion = torch.cat((pred_fusion, out_fusion.data), 0)
                #loss sum
                loss_tensor = loss_img + loss_roi + loss_fusion
                val_loss.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : image loss ={}, roi loss ={}, fusion loss = {}, train loss = {}'
                                .format(epoch+1, batch_idx+1, float('%0.6f'%loss_img.item()), float('%0.6f'%loss_roi.item()),
                                float('%0.6f'%loss_fusion.item()), float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
                
                loss_img_all.append(loss_img.item())
                loss_roi_all.append(loss_roi.item())
                loss_fusion_all.append(loss_fusion.item())
        #evaluation       
        AUROCs_img = np.array(compute_AUCs(gt, pred_img)).mean()
        AUROCs_roi = np.array(compute_AUCs(gt, pred_roi)).mean()
        AUROCs_fusion = np.array(compute_AUCs(gt, pred_fusion)).mean()
        print("\r Eopch: %5d validation loss = %.6f, Validataion AUROC image=%.4f roi=%.4f fusion=%.4f" 
              % (epoch + 1, np.mean(val_loss), AUROCs_img, AUROCs_roi, AUROCs_fusion)) 

        logger.info("\r Eopch: %5d validation loss = %.4f, image loss = %.4f,  roi loss =%.4f fusion loss =%.4f" 
                     % (epoch + 1, np.mean(val_loss), np.mean(loss_img_all), np.mean(loss_roi_all), np.mean(loss_fusion_all))) 
        #save checkpoint
        if AUROC_best < AUROCs_fusion:
            AUROC_best = AUROCs_fusion
            #torch.save(model.module.state_dict(), CKPT_PATH)
            torch.save(model_img.state_dict(), config['CKPT_PATH'] +  'img_model.pkl') #Saving torch.nn.DataParallel Models
            torch.save(model_roi.state_dict(), config['CKPT_PATH'] + 'roi_model.pkl')
            torch.save(model_fusion.state_dict(), config['CKPT_PATH'] + 'fusion_model.pkl')
            print(' Epoch: {} model has been already save!'.format(epoch+1))
    
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    #dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model_img = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True, is_roi=False).cuda()
        CKPT_PATH = config['CKPT_PATH']  +'img_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_img.load_state_dict(checkpoint) #strict=False
        print("=> loaded Image model checkpoint: "+CKPT_PATH)

        model_roi = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True, is_roi=True).cuda()
        CKPT_PATH = config['CKPT_PATH'] + 'roi_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_roi.load_state_dict(checkpoint) #strict=False
        print("=> loaded ROI model checkpoint: "+CKPT_PATH)

        model_fusion = FusionClassifier(input_size=2048, output_size=N_CLASSES).cuda()
        CKPT_PATH = config['CKPT_PATH'] + 'fusion_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_fusion.load_state_dict(checkpoint) #strict=False
        print("=> loaded Fusion model checkpoint: "+CKPT_PATH)

        roigen = ROIGenerator() #region generator

    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor().cuda()
    pred_img = torch.FloatTensor().cuda()
    pred_roi = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()
    # switch to evaluate mode
    model_img.eval() #turn to test mode
    model_roi.eval()
    model_fusion.eval()
    cudnn.benchmark = True
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            gt = torch.cat((gt, label.cuda()), 0)
            #image-level
            var_image = torch.autograd.Variable(image).cuda()
            #var_label = torch.autograd.Variable(label).cuda()
            conv_fea_img, fc_fea_img, out_img = model_img(var_image)#forward
            pred_img = torch.cat((pred_img, out_img.data), 0)
            #ROI-level
            roi = roigen.ROIGeneration(conv_fea_img)
            var_roi = torch.autograd.Variable(roi).cuda()
            out_roi = model_roi(var_roi)
            pred_roi = torch.cat((pred_roi, out_roi.data), 0)
            #Fusion
            fc_fea_fusion = torch.cat((fc_fea_img,var_roi), 1)
            var_fusion = torch.autograd.Variable(fc_fea_fusion).cuda()
            out_fusion = model_fusion(var_fusion)
            pred_fusion = torch.cat((pred_fusion, out_fusion.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #for evaluation
    AUROC_img = compute_AUCs(gt, pred_img)
    AUROC_avg = np.array(AUROC_img).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_img[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    AUROC_roi = compute_AUCs(gt, pred_roi)
    AUROC_avg = np.array(AUROC_roi).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_roi[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    AUROC_fusion = compute_AUCs(gt, pred_fusion)
    AUROC_avg = np.array(AUROC_fusion).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROC_fusion[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    #Evaluating the threshold of prediction
    #thresholds = compute_ROCCurve(gt, pred_fusion)
    #print(thresholds)

def BoxTest():
    print('********************load data********************')
    dataloader_bbox = get_bbox_dataloader(batch_size=1, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CXRNet':
        model_img = CXRClassifier(num_classes=N_CLASSES, is_pre_trained=True, is_roi=False).cuda()
        CKPT_PATH = config['CKPT_PATH'] + 'img_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model_img.load_state_dict(checkpoint) #strict=False
        print("=> loaded Image model checkpoint: "+CKPT_PATH)
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
            #if batch_idx != 963: continue
            var_image = torch.autograd.Variable(image).cuda()
            conv_fea_img, fc_fea_img, out_img = model_img(var_image) #get feature maps
            """
            logit = out_img.cpu().data.numpy().squeeze() #predict
            idxs = []
            for i in range(N_CLASSES):
                if logit[i] > thresholds[i]: #diffrent diseases vary in threshold
                    idxs.append(i)
            """
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
    #Train() #for training
    Test() #for test
    #BoxTest()

if __name__ == '__main__':
    main()