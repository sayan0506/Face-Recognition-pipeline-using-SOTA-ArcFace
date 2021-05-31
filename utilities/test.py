from pathlib import Path
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import FaceDataset as fd

# data pipeline
# helps to create dict where, use keys as atribute
from easydict import EasyDict as edict

from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile, ImageDraw
from torch.utils.data import Dataset, DataLoader

import model

# ensures loading truncated images 
ImageFile.LOAD_TRUNCATED_IMAGES = True

# model build
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
import yaml
# model training
from torch import optim
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import argparse
import inference_learner as il


class test_pipeline(object):
  def __init__(self, dataset_path, pretrained_save_path, csv_path):
    # config edict dictionary initialize
    self.config = edict()
    self.config.device = self.device_alloc()
    self.config.imgs_folder = dataset_path
    # identity list
    self.identity_list = ['chris_evans', 'chris_hemsworth', 'mark_ruffalo', 'robert_downey_jr', 'scarlett_johansson']
    # identity path
    self.identity_path = [os.path.join(self.config.imgs_folder, identity) for identity in self.identity_list]
    
    # configure identity_list 
    self.config.identity_list = self.identity_list  
    
    # input image size for the model 
    self.config.input_size = (112,112)
    # configure transform
    self.config.transform = trans.Compose([
                                      trans.RandomHorizontalFlip(), # random horizonttal flip of faces
                                      trans.ToTensor(), # convert img to torch tensor
                                      trans.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), # normalize
                                      trans.Resize(self.config.input_size, interpolation=trans.InterpolationMode.BICUBIC)
      ])

    # initializing data_mode as 'ms1m' dataset
    self.config.data_mode = 'ms1m'
    # initializing batch_size to 64
    self.config.batch_size = 64
    # pin_memory status is set to True, which enables to load samples of data to device(GPU), spped-up the training
    self.config.pin_memory = True
    # initializing the number of workers to 3
    self.config.num_workers = 3
    # embedding size
    self.config.embedding_size = 512
    self.config.use_mobilefacenet = True

    self.config.valid_path = os.path.join(csv_path, 'valid.csv')
    self.config.test_path = os.path.join(csv_path, 'test.csv')    
    self.config.best_model = '/content/saved_model/model_step_41_loss_0.15294727683067322.path'
    # default fixed threshold value 1.65
    self.config.threshold = 1.65




  def device_alloc(self):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    assert device == torch.device('cuda:0')
    print(f'Device info\n{torch.cuda.get_device_properties(0)}')
    
    return device
  

  def get_dataset(self, df):
    # fetch and transform the images using torchvision transforms to create dataset
    # from the image folder
    ds = fd.FaceDataset(df, self.config.imgs_folder, self.config.identity_list, transform= self.config.transform)
    class_num = len(self.config.identity_list) # total class is the index of last folder + 1
    return ds, class_num

  # define the train_loader function, where we can pass dataset mode(as 'vgg' or 'ms1m') through configuration dictionary
  def get_data_loader(self, ds, class_num):
    if self.config.data_mode in ['ms1m','vgg']:
      # create train_loader, by deafult shuffles the datapoints
      data_loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle = True, pin_memory = self.config.pin_memory,
                                num_workers = self.config.num_workers)
      return data_loader, class_num
  
  def inference_result(self, src_embed_pair, dst_embed_pair, config):
    result = []
    # traverse through all the test enchoding pair
    for embed, label in dst_embed_pair:
      dist = []
      # calculate dist between validation and test embeddings
      for val_embed, val_label in src_embed_pair:
        diff = val_embed - embed
        dist.append(torch.sum(torch.pow(diff, 2)))
      
      # checks the index whose dist is minimum
      if np.min(dist) < config.threshold:
        min_idx = np.argmin(dist)
        result.append(int(src_embed_pair[min_idx][1] == label))

      else:
        result.append(0)
    
    print(f'Test performance accuracy {(np.sum(result)/len(result))*100}%')

  def gen_embedding(self, data_loader, model):
    
    # stores list of actual label, embedding pair
    embedding_pair = []
    for imgs, labels in tqdm(iter(data_loader)):
        imgs = imgs.to(self.config.device)
        inference_model = model
        embeddings = inference_model(imgs)
    
    # merging labels to them
    for i, label in enumerate(labels):
      pair = tuple(list([embeddings[i], label]))
      embedding_pair.append(pair)
    
    return embedding_pair

  def run_pipeline(self):
    
    self.config.valid_df = pd.read_csv(self.config.valid_path)
    self.config.test_df = pd.read_csv(self.config.test_path)
    
    
    # create validation dataset
    valid_ds, valid_class_num = self.get_dataset(self.config.valid_df)
    
    targets = [valid_ds.identity.index(id) for id in valid_ds.identity]
    print(f'\nValidation Dataset info {valid_ds},\ncontains {valid_class_num} different identities!')
    print(f'Classes available in the dataset \n{valid_ds.identity},\nhaving class ids {targets} respectively!')
    print(f'Length of the dataset {len(valid_ds)}')

    # create test dataset
    test_ds, test_class_num = self.get_dataset(self.config.test_df)
    print(f'\nTest Dataset info {test_ds},\ncontains {test_class_num} different identities!')
    print(f'Classes available in the dataset \n{test_ds.identity},\nhaving class ids {targets} respectively!')
    print(f'Length of the dataset {len(test_ds)}')
    
    # valid_loader
    valid_loader, _ = self.get_data_loader(valid_ds, valid_class_num)
    # test_loader
    test_loader, _ = self.get_data_loader(test_ds, test_class_num)

    print('\nTrain, valid, test dataloader have created!')

    # call model object
    mobilefacenet_model = model.MobileFaceNet(embedding_size= self.config.embedding_size).to(self.config.device)
    # load the model for inference/test
    inference_learn = il.inference_face_learner(self.config, mobilefacenet_model, valid_loader, valid_class_num)
    # source embedding pair
    src_embed_pair = self.gen_embedding(valid_loader, inference_learn.model)
    # destination embedding pair
    dst_embed_pair = self.gen_embedding(test_loader, inference_learner.model)

    self.inference_result(src_embed_pair, dst_embed_pair, self.config)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "ArcFace Face Recognition Pipeline") 
  
  parser.add_argument('-dataset', '--dataset_path', default = ".", type = str,
                      help = 'Path of the dataset')

  parser.add_argument('-model', '--trained_save_path', default = ".", type = str,
                      help = 'pre-trained model folder, only mention the folder')
  
  parser.add_argument('-df', '--csv_save_path', default = ".", type = str,
                      help = 'path of valid, test csv folder')

  args = parser.parse_args()
  
  dataset_path = args.dataset_path
  pretrained_save_path = args.trained_save_path
  csv_path = args.csv_save_path

  # define the config
  test_p = test_pipeline(dataset_path, pretrained_save_path, csv_path)   
  # run the pipeline
  test_p.run_pipeline()





