import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, df, root_dir, identity_list, transform = None):
      self.df = df
      self.dir = root_dir
      self.transform = transform
      self.identity = identity_list
    
    def __len__(self):
      return self.df.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()
      img_name = os.path.join(self.dir, self.df['Label'].iloc[idx], self.df['Image'].iloc[idx])
      sample = [Image.open(img_name), int(self.identity.index(self.df['Label'].iloc[idx]))]

      if self.transform:
        sample[0] = self.transform(sample[0])
      
      sample = tuple(sample)

      return sample
