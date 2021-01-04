import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import os
from os import path
import numpy as np
from PIL import Image
import csv
random.seed(42)

class ImageDataset(Dataset):
    def __init__(self, root,
            transform=transforms.Compose([transforms.ToTensor()]),
            img_size=128, file_extn="jpg"):
        
        super(ImageDataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.imgs = os.listdir(self.root)
        self.transform = transform
        self.img_size = img_size
        self.file_extn = file_extn
        
    def __len__(self):
        return len(self.imgs)

    def pil_loader(self, path, img_size, mode='RGBA'):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if img_size != img.size[0]:
                img = img.resize((img_size, img_size))
            return img.convert(mode)

    def __getitem__(self, index):
        img_path = path.join(self.root, self.imgs[index])
        rgba = self.pil_loader(
                path.join(self.root, self.imgs[index]),
                img_size=self.img_size)

        if self.transform is not None:
            rgba = self.transform(rgba)
        img = rgba[:3]
        return img
