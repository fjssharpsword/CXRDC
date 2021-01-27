import os
import torchvision.transforms as transforms
#config for training
config = {
            'CKPT_PATH': '/data/pycode/CXR-IRNet/model/',
            'log_path':  '/data/pycode/CXR-IRNet/log/',
            'img_path': '/data/pycode/CXR-IRNet/imgs/',
            'CUDA_VISIBLE_DEVICES': '7', #"0,1,2,3,4,5,6,7"
            'MAX_EPOCHS': 3, #30, 
            'BATCH_SIZE': 64, 
            'TRAN_SIZE': 256,
            'TRAN_CROP': 224
         } 

#config for dataset
PATH_TO_IMAGES_DIR = '/data/fjsdata/NIH-CXR/images/images/'
PATH_TO_TRAIN_COMMON_FILE = '/data/pycode/CXR-IRNet/dataset/train.txt'
PATH_TO_VAL_COMMON_FILE = '/data/pycode/CXR-IRNet/dataset/val.txt'
PATH_TO_TEST_COMMON_FILE = '/data/pycode/CXR-IRNet/dataset/test.txt'
PATH_TO_BOX_FILE = '/data/pycode/CXR-IRNet/dataset/fjs_BBox.csv'
PATH_TO_TRAIN_VAL_BENCHMARK_FILE = '/data/pycode/CXR-IRNet/dataset/bm_train_val.csv'
PATH_TO_TEST_BENCHMARK_FILE = '/data/pycode/CXR-IRNet/dataset/bm_test.csv'
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = len(CLASS_NAMES)
transform_seq_test = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_train = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   #transforms.RandomCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

#config for project
# threshold for each disease
classes_threshold_common =   {0: 0.1138, #'Atelectasis',
                              1: 0.0200, #'Cardiomegaly',
                              2: 0.0942, #'Effusion', 
                              3: 0.1963, #'Infiltration',
                              4: 0.0202, #'Mass', 
                              5: 0.0367, #'Nodule', 
                              6: 0.0249, #'Pneumonia',
                              7: 0.0160, #'Pneumothorax',
                              8: 0.0308, #'Consolidation',
                              9: 0.0206, #'Edema',
                              10: 0.0126, #'Emphysema',
                              11: 0.0184, #'Fibrosis',
                              12: 0.0330, #'Pleural_Thickening',
                              13: 0.0037  #'Hernia'
                              }
classes_threshold_benchmark ={0: 0.1141, #'Atelectasis',
                              1: 0.0238, #'Cardiomegaly',
                              2: 0.1425, #'Effusion', 
                              3: 0.2306, #'Infiltration',
                              4: 0.0260, #'Mass', 
                              5: 0.0394, #'Nodule', 
                              6: 0.0300, #'Pneumonia',
                              7: 0.0308, #'Pneumothorax',
                              8: 0.0496, #'Consolidation',
                              9: 0.0400, #'Edema',
                              10: 0.0130, #'Emphysema',
                              11: 0.0175, #'Fibrosis',
                              12: 0.0398, #'Pleural_Thickening',
                              13: 0.0033  #'Hernia'
                              }