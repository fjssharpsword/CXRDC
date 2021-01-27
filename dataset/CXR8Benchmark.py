import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
import re

#define myself
from config import *
"""
Dataset: Chest X-Ray8
https://www.kaggle.com/nih-chest-xrays/data
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
1) 112,120 X-ray images with disease labels from 30,805 unique patients
2）Label:['Atelectasis', 'Cardiomegaly', 'Effusion','Infiltration', 'Mass', 'Nodule', 'Pneumonia', \
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
"""
#generate dataset 
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f:
                    items = line.strip().split(',') 
                    image_name= items[0]#.split('/')[1]
                    label = list(items[1].replace(' ', ''))[1:15]
                    label = [int(eval(i)) for i in label]
                    image_name = os.path.join(path_to_img_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

        """
        #statistics of dataset
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        labels_np = np.array(labels)
        multi_dis_num = 0
        for i in range(len(CLASS_NAMES)):
            num = len(np.where(labels_np[:,i]==1)[0])
            multi_dis_num = multi_dis_num + num
            print('Number of {} is {}'.format(CLASS_NAMES[i], num))
        print('Number of Multi Finding is {}'.format(multi_dis_num))

        norm_num = (np.sum(labels_np, axis=1)==0).sum()
        dis_num = (np.sum(labels_np, axis=1)!=0).sum()
        assert norm_num + dis_num==len(labels)
        print('Number of No Finding is {}'.format(norm_num))
        print('Number of Finding is {}'.format(dis_num))
        print('Total number is {}'.format(len(labels)))
        """

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=[PATH_TO_TEST_BENCHMARK_FILE], transform=transform_seq_test)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

#for cross-validation
def get_train_val_dataloader(batch_size, shuffle, num_workers, split_ratio=0.1):
    dataset_train_full = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                         path_to_dataset_file=[PATH_TO_TRAIN_VAL_BENCHMARK_FILE], transform=transform_seq_train)

    val_size = int(split_ratio * len(dataset_train_full))
    train_size = len(dataset_train_full) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_train_full, [train_size, val_size])

    data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers, pin_memory=True)
    return data_loader_train, data_loader_val

def preprocess():
    #generating train set and test set
    meta_data = pd.read_csv("./Dataset/Data_Entry_2017_v2020.csv" , sep=',') #detailed information of images
    # define dummy labels for one hot encoding - simplifying to 15 primary classes (Including. No Finding)
    dummy_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion','Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', \
                    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'] 
    for label in dummy_labels:
        meta_data[label] = meta_data['Finding Labels'].map(lambda result: 1 if label in result else 0)
    meta_data['target_vector'] = meta_data.apply(lambda target: [target[dummy_labels].values], 1).map(lambda target: target[0])
    print('Dataset statistic, records: %d, fields: %d'%(meta_data.shape[0], meta_data.shape[1]))
    print(meta_data.columns.values.tolist())

    train_val_list_path = "./Dataset/train_val_list.txt" 
    test_list_path = "./Dataset/test_list.txt" 

    with open(train_val_list_path, "r") as f:
        train_list = [ i.strip() for i in f.readlines()]
    with open(test_list_path, "r") as f:
        test_list = [ i.strip() for i in f.readlines()]

    def get_labels(pic_id):
        labels = meta_data.loc[meta_data["Image Index"]==pic_id,"target_vector"]
        return np.array(labels.tolist()[0])

    train_y = []
    for train_id in train_list:
        train_y.append(get_labels(train_id))
    test_y = []
    for test_id in test_list:
        test_y.append(get_labels(test_id))

    df_train = pd.DataFrame({'image_index':train_list,'target_vector':train_y})
    print('Trainset statistic, records: %d, fields: %d'%(df_train.shape[0], df_train.shape[1]))
    df_test = pd.DataFrame({'image_index':test_list,'target_vector':test_y})
    print('Testset statistic, records: %d, fields: %d'%(df_test.shape[0], df_test.shape[1]))
    df_train.to_csv('./Dataset/bm_train_val.csv', index=False, header=False)
    df_test.to_csv('./Dataset/bm_test.csv', index=False, header=False)

if __name__ == "__main__":

    #preprocess()

    #for debug   
    data_loader = get_test_dataloader(batch_size=10, shuffle=False, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader):
        print(label.shape)
        break