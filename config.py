import os
import torchvision.transforms as transforms
#config for training
config = {
            'CKPT_PATH': '/data/pycode/CXRDC/model/',
            'log_path':  '/data/pycode/CXRDC/log/',
            'img_path': '/data/pycode/CXRDC/imgs/',
            'CUDA_VISIBLE_DEVICES': '7', #"0,1,2,3,4,5,6,7"
            'MAX_EPOCHS': 30, 
            'BATCH_SIZE': 16, 
            'TRAN_SIZE': 256,
            'TRAN_CROP': 224
         } 

#config for dataset
PATH_TO_IMAGES_DIR = '/data/fjsdata/NIH-CXR/images/images/'
PATH_TO_BOX_FILE = '/data/pycode/CXRDC/dataset/fjs_BBox.csv'
PATH_TO_TRAIN_VAL_BENCHMARK_FILE = '/data/pycode/CXRDC/dataset/bm_train_val.csv'
PATH_TO_TEST_BENCHMARK_FILE = '/data/pycode/CXRDC/dataset/bm_test.csv'
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