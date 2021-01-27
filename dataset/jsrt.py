import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
from PIL import Image
import PIL.ImageOps 
import cv2
from scipy.io import loadmat
"""
Dataset: JSRT CXR
https://www.kaggle.com/raddar/nodules-in-chest-xrays-jsrt
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        mask_names = []
        labels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f: 
                    items = line.strip().split(',') 
                    image_name = os.path.join(path_to_img_dir, 'images', items[0])
                    image_names.append(image_name)
                    mask_left = os.path.join(path_to_img_dir,'masks/left_lung/', os.path.splitext(items[0])[0]+'.gif')
                    mask_right = os.path.join(path_to_img_dir,'masks/right_lung/', os.path.splitext(items[0])[0]+'.gif')
                    mask_heart = os.path.join(path_to_img_dir,'masks/heart/', os.path.splitext(items[0])[0]+'.gif')
                    mask_names.append([mask_left, mask_right, mask_heart])
                    
        self.image_names = image_names
        self.mask_names = mask_names
        self.transform_seq_image = transforms.Compose([
            transforms.Resize((256,256)),#256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
        self.transform_seq_mask = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224)
            ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        try:
            image_name = self.image_names[index]
            image = Image.open(image_name).convert('RGB')
            image = self.transform_seq_image(image)
            
            """
            mask_name = self.mask_names[index]
            mask_left = np.asarray(Image.open(mask_name[0]))
            mask_right = np.asarray(Image.open(mask_name[1]))
            mask_heart = np.asarray(Image.open(mask_name[2]))
            mask = Image.fromarray(mask_left + mask_right + mask_heart)
            #mask.save('/data/pycode/CXR-IRNet/imgs/jsrt_msk.jpeg',"JPEG", quality=95, optimize=True, progressive=True)
            mask = torch.FloatTensor(np.array(self.transform_seq_mask(mask)))
            #turn to 0-0, else=1, 255=1
            mask = torch.where(mask!=0, torch.full_like(mask, 1), mask)
            """
            mask_name = self.mask_names[index]
            mask_left = torch.FloatTensor(np.array(self.transform_seq_mask(Image.open(mask_name[0]))))
            mask_left = torch.where(mask_left!=0, torch.full_like(mask_left, 1), mask_left)
            mask_right = torch.FloatTensor(np.array(self.transform_seq_mask(Image.open(mask_name[1]))))
            mask_right = torch.where(mask_right!=0, torch.full_like(mask_right, 1), mask_right)
            mask_heart = torch.FloatTensor(np.array(self.transform_seq_mask(Image.open(mask_name[2]))))
            mask_heart = torch.where(mask_heart!=0, torch.full_like(mask_heart, 1), mask_heart)

        except Exception as e:
            print("Unable to read file. %s" % e)
        
        return image, mask_left, mask_right, mask_heart

    def __len__(self):
        return len(self.image_names)

#config 
PATH_TO_IMAGES_DIR = '/data/fjsdata/JSRT-CXR/'
PATH_TO_TRAIN_FILE = '/data/pycode/CXR-IRNet/dataset/jsrt_list.txt'

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,path_to_dataset_file=[PATH_TO_TRAIN_FILE])
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)#drop_last=True
    return data_loader_train

if __name__ == "__main__":
  
    #for debug   
    data_loader_train = get_train_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, mask_left, mask_right, mask_heart) in enumerate(data_loader_train):
        print(batch_idx)
        print(image.shape)
        print(mask_left.shape)
        break
    