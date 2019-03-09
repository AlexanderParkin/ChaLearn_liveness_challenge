import torch.utils.data as data
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os

import torch.utils.data as data

def rgb_loader(path):
    return Image.open(path)


class ImageListDataset(data.Dataset):
    """
    Builds a dataset based on a list of images.
    data_root - image path prefix
    data_list - annotation list location
    """
    def __init__(self, data_root, data_list, transform=None):
        self.data_root = data_root
        #self.df = pd.read_csv(data_list)
        self.df = pd.read_csv(data_list)
        if 'label' not in self.df.columns:
            self.df['label'] = -1
        self.transform = transform
        self.loader = rgb_loader

    def __getitem__(self, index):

        path = self.data_root + self.df.path.iloc[index]
        img = self.loader(path)

        label = self.df.label.iloc[index]

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.df)
