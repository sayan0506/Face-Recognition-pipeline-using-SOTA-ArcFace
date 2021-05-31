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
import face_learner as fcl

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


class train_pipeline(object):
  def __init__(self, dataset_path, pretrained_save_path):
    # config edict dictionary initialize
    self.config = edict()
    self.config.device = self.device_alloc()
    self.config.imgs_folder = dataset_path
    # identity list
    self.identity_list = ['chris_evans', 'chris_hemsworth', 'mark_ruffalo', 'robert_downey_jr', 'scarlett_johansson']
    # identity path
    self.identity_path = [os.path.join(self.config.imgs_folder, identity) for identity in self.identity_list]
    # image dataframe
    self.img_df = pd.DataFrame(columns = ['Image', 'Label'])

    # label dataframe
    self.label_df = pd.DataFrame(columns = self.identity_list)
    
    self.config.csv_path = 'csv_data_files'

    if not os.path.isdir(self.config.csv_path):
      os.mkdir(self.config.csv_path)

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
    # configure mobilefacenet status
    self.config.use_mobilefacenet = True

    # configure milestones
    self.config.milestones = [12,15,18]

    # configure momentum
    self.config.momentum = 0.9

    # configure learning rate
    self.config.lr = 1e-03

    # loss fn
    self.config.ce_loss = CrossEntropyLoss()

    # mobilefacenet load model path
    self.config.pretrained_save_path = pretrained_save_path

    self.config.save_path = 'saved_model'

    if not os.path.isdir(self.config.save_path):
      os.mkdir(self.config.save_path)

    print(f'Best validated model will be saved in "{self.config.save_path}" folder!')





  def device_alloc(self):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    assert device == torch.device('cuda:0')
    print(f'Device info\n{torch.cuda.get_device_properties(0)}')
    
    return device
  
  # create image and label dataframe
  def create_dataframe(self):    
    print('\n[INFO] Loading images ...')
    # total sample count
    sample_count = 0
    for sr, id_path in enumerate(self.identity_path):
      print(f'[INFO] Processing {self.identity_list[sr]}')
      sample_list = os.listdir(id_path)

      for id_sample in sample_list:
        # removing the .DS_store files from list, which contains the folder infos
        if id_sample == '.DS_store':
          sample_list.remove(id_sample)
      # categorical count
      count = 0
      for id_sample in sample_list:
        id_sample_path = os.path.join(id_path, id_sample)
        if id_sample_path.endswith(".jpg") == True or id_sample_path.endswith(".JPG") == True or id_sample_path.endswith(".png") == True or id_sample_path.endswith(".PNG") == True:
          self.img_df.loc[sample_count,'Image'], self.img_df.loc[sample_count,'Label'] = id_sample, self.identity_list[sr]
          #self.img_df.loc[sample_count,'Shape'] = img.shape
          count += 1
          sample_count += 1

      self.label_df.loc[0, self.identity_list[sr]] = count
      print(count)
      # checks whether all the samples are loaded successfully or not to img_df
      assert len(sample_list) == count

    print(f'Data distribution of different identities\n{self.label_df}')
    print(f'Image dataframe\n{self.img_df}')

  # train-test-split
  def train_valid_test_split(self):
    
    # validation split
    val_ratio = 0.10
    # test split 50% of validation_main data
    test_ratio = 0.5

    self.config.split_ratio = [0.9,0.05,0.05]

    # image_ids
    img_ids = self.img_df.loc[:,'Image']

    # image_ids
    label_ids = self.img_df.loc[:,'Label']
    
    # train df
    train_df = pd.DataFrame(columns=['Image','Label'])
    # validation main df
    valid_main_df = pd.DataFrame(columns=['Image','Label'])
    # validation df
    valid_df = pd.DataFrame(columns=['Image','Label'])
    # test df
    test_df = pd.DataFrame(columns=['Image','Label'])

    # train, validation_main split
    train_df['Image'], valid_main_df['Image'], train_df['Label'], valid_main_df['Label'] = train_test_split(img_ids, label_ids,
                                                                                                      test_size = val_ratio,
                                                                                                      random_state = 28,
                                                                                                      stratify = label_ids,
                                                                                                      shuffle = True
                                                                                                      )

    # validation, test split
    valid_df['Image'], test_df['Image'], valid_df['Label'], test_df['Label'] = train_test_split(valid_main_df['Image'], valid_main_df['Label'],
                                                                                                      test_size = test_ratio,
                                                                                                      random_state = 28,
                                                                                                      stratify = valid_main_df['Label'],
                                                                                                      shuffle = True
                                                                                                      )
    
    print('Stratified split distribution-')
    print(train_df['Label'].value_counts())
    print(valid_df['Label'].value_counts())
    print(test_df['Label'].value_counts())
    
    # configure train, valid, test
    self.config.train_df = train_df
    self.config.valid_df = valid_df
    self.config.test_df = test_df

    self.config.train_df.to_csv(os.path.join(self.config.csv_path, 'train.csv'))
    self.config.valid_df.to_csv(os.path.join(self.config.csv_path, 'valid.csv'))
    self.config.test_df.to_csv(os.path.join(self.config.csv_path, 'test.csv'))

    print(f'Train, Valid, Test dataframes are stored in csv to {self.config.csv_path}')
 
  
  
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
  

  def run_pipeline(self):
    # dataframe
    self.create_dataframe()
    # data split
    self.train_valid_test_split()

    # create train dataset
    train_ds, train_class_num = self.get_dataset(self.config.train_df)
    targets = [train_ds.identity.index(id) for id in train_ds.identity]

    print(f'\nTrain Dataset info {train_ds},\ncontains {train_class_num} different identities!')
    print(f'Classes available in the dataset \n{train_ds.identity},\nhaving class ids {targets} respectively!')
    print(f'Length of the dataset {len(train_ds)}')

    # create validation dataset
    valid_ds, valid_class_num = self.get_dataset(self.config.valid_df)
    print(f'\nValidation Dataset info {valid_ds},\ncontains {valid_class_num} different identities!')
    print(f'Classes available in the dataset \n{valid_ds.identity},\nhaving class ids {targets} respectively!')
    print(f'Length of the dataset {len(valid_ds)}')

    # create test dataset
    test_ds, test_class_num = self.get_dataset(self.config.test_df)
    print(f'\nTest Dataset info {test_ds},\ncontains {test_class_num} different identities!')
    print(f'Classes available in the dataset \n{test_ds.identity},\nhaving class ids {targets} respectively!')
    print(f'Length of the dataset {len(test_ds)}')
    
    # train_loader
    train_loader, _ = self.get_data_loader(train_ds, train_class_num)
    # valid_loader
    valid_loader, _ = self.get_data_loader(valid_ds, valid_class_num)
    # test_loader
    test_loader, _ = self.get_data_loader(test_ds, test_class_num)

    print('\nTrain, valid, test dataloader have created!')

    # call model object
    mobilefacenet_model = model.MobileFaceNet(embedding_size= self.config.embedding_size).to(self.config.device)

    print(f'\nMobileFcaenet Model create\nModel summary\n{mobilefacenet_model}')

    learner = fcl.face_learner(self.config, mobilefacenet_model, train_loader, train_class_num, valid_loader, valid_class_num)
    learner.load_state(self.config, 'mobilefacenet.pth', from_save_folder = True, model_only = True)
    print('Mobilenet pre-trained loaded successfully!')
    learner.model.train()

    learner.model.train()
    running_loss = 0
    # stores loss of each step
    loss_bank = []

    epochs = 20

    for e in range(epochs):
      print('epoch {} started'.format(e))
      # at each milestone learning rate is changes according to the scheduler
      if e == learner.milestones[0]:
          learner.schedule_lr()
      if e == learner.milestones[1]:
          learner.schedule_lr()      
      if e == learner.milestones[2]:
          learner.schedule_lr()                                 
      
      for imgs, labels in tqdm(iter(learner.loader)):
          imgs = imgs.to(self.config.device)
          labels = labels.to(self.config.device)
          learner.optimizer.zero_grad()
          embeddings = learner.model(imgs)
          thetas = learner.head(embeddings, labels)
          loss = self.config.ce_loss(thetas, labels)
          loss.backward()
          running_loss += loss.item()
          learner.optimizer.step()
          learner.step += 1
          print(f'Train Loss of step {learner.step} is  {loss}')
          
          val_loss = 0
          # create validation loss
          for val_imgs, val_labels in tqdm(iter(learner.val_loader)):
            val_imgs = val_imgs.to(self.config.device)
            val_labels = val_labels.to(self.config.device)
            learner.optimizer.zero_grad()
            val_embeddings = learner.model(val_imgs)
            val_thetas = learner.head(val_embeddings, val_labels)
            val_loss += self.config.ce_loss(val_thetas, val_labels)
            
          print(f'Validation Loss of step {learner.step} is {val_loss}')

          if len(loss_bank):
            if loss_bank[-1]>val_loss:
              learner.save_state(self.config, val_loss)
              print(f'Saving model at epoch {e}, step {learner.step}') 
              loss_bank = [] 
              
              # if current validation loss is min, then store it, else don't store
              loss_bank.append(val_loss)

          else:
            loss_bank.append(val_loss)

    print(f'\nMin validation loss - {loss_bank[0]}')    
    print(f'\nBest model based on validation data result - {learner.save_validated_model}')
    self.config.best_model = learner.save_validated_model            

    with open('config.yml', 'w') as outfile:
      yaml.dump(self.config, outfile, default_flow_style = False)

    print("Configuration is stored in 'config.yml!'")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "ArcFace Face Recognition Pipeline") 
  
  parser.add_argument('-dataset', '--dataset_path', default = ".", type = str,
                      help = 'Path of the dataset')

  parser.add_argument('-model', '--pretrained_save_path', default = ".", type = str,
                      help = 'pre-trained model folder, only mention the folder')
  
  args = parser.parse_args()
  
  dataset_path = args.dataset_path
  pretrained_save_path = args.pretrained_save_path

  # define the config
  train_p = train_pipeline(dataset_path, pretrained_save_path)   
  # run the pipeline
  train_p.run_pipeline()





