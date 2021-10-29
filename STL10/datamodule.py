from torchvision import datasets, transforms

import torch
import random
import torch.utils.data as data
import pytorch_lightning as pl
import numpy as np
from PIL import Image, PILLOW_VERSION
import h5py

class STL10Dataset(data.Dataset):
    def __init__(self, 
                 data_dir,
                 fillcolor=(128, 128, 128),
                 resample=Image.BILINEAR,
                 pre_transform=None,
                 split_type="unsupervised"):
           
        self.data_dir = data_dir
        self.fillcolor = fillcolor
        self.resample = resample
        self.split_type = split_type
        self.pre_transform = pre_transform
        
        #self.dataset = datasets.ImageFolder(self.data_dir, self.pre_transform)
        if self.split_type == "unsupervised":
            self.dataset = datasets.STL10(root=self.data_dir, split="unlabeled", transform=self.pre_transform, download=False)
        elif self.split_type == "train":
            self.dataset = datasets.STL10(root=self.data_dir, split="train", transform=self.pre_transform, download=False)
        elif self.split_type == "test":
            self.dataset = datasets.STL10(root=self.data_dir, split="test", transform=self.pre_transform, download=False)
        elif self.split_type == "valid":
            self.dataset = datasets.STL10(root=self.data_dir, split="train", folds=1, transform=self.pre_transform, download=False)
        
        mean_pix = [0.4914, 0.4822, 0.4465] 
        std_pix = [0.2023, 0.1994, 0.2010]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        
        self.matrix_transform = transforms.Compose([
            transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015))
        ])
        
        with h5py.File('../homography.h5', 'r') as hf:
            self.homography = hf['homography'][:]
        
        
    def __getitem__(self, index):
        if self.split_type == "unsupervised":
            img0, _ = self.dataset[index]
            
            width, height = img0.size
            center = (img0.size[0] * 0.5 + 0.5, img0.size[1] * 0.5 + 0.5)

            coeffs = self.homography[random.randint(0,499999)]

            kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
            img1 = img0.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)

            ori_img = self.transform(img0)
            warped_img = self.transform(img1)

            coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)
            coeffs = self.matrix_transform(coeffs)
            coeffs = coeffs.view(8)

            return ori_img, warped_img, coeffs
            
        else:
            img, categorical_label = self.dataset[index]
            img = self.transform(img)
            return img, categorical_label

    def __len__(self):
        return len(self.dataset)


class DataModuleSTL10(pl.LightningDataModule):
    def __init__(self,
                 unsupervised_dir=None,
                 train_dir=None,
                 test_dir=None,
                 valid_dir=None,
                 unsupervised=True,
                 batch_size=512):

        super().__init__()
        
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.unsupervised_dir = unsupervised_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.valid_dir = valid_dir

    def setup(self, stage=None): 
        if stage == "fit" or stage is None:
                                
            transform_list = [transforms.Resize(32),
                              transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip()]
           
            
            pre_transform = transforms.Compose(transform_list)
           
            transform_list2 = [transforms.Resize(32)]
            pre_transform2 = transforms.Compose(transform_list2)
            
            if self.unsupervised:
                self.train_stl10 = STL10Dataset(data_dir=self.unsupervised_dir, split_type="unsupervised", pre_transform=pre_transform)
                
            else:
                #self.train_stl10 = STL10Dataset(data_dir=self.train_dir, split_type="train", pre_transform=pre_transform)
                #self.valid_stl10 = STL10Dataset(data_dir=self.valid_dir, split_type="valid", pre_transform=pre_transform2)
                
                
                stl10 = STL10Dataset(data_dir=self.train_dir, split_type="train", pre_transform=pre_transform)
                
                size = len(stl10)

                valid_size = int(0.1 * size)
                train_size = size - valid_size
                seed=42
                generator=torch.Generator().manual_seed(seed)
                self.train_stl10, self.valid_stl10 = torch.utils.data.random_split(stl10, [train_size, valid_size], generator=generator)
                
        if stage == "test" or stage is None:
            if not self.unsupervised:
                transform_lst = [transforms.Resize(32)]
                pre_trsf = transforms.Compose(transform_lst)
                self.test_stl10 = STL10Dataset(data_dir=self.test_dir, split_type="test", pre_transform=pre_trsf)
                

    def train_dataloader(self):
        return data.DataLoader(dataset=self.train_stl10,
                            batch_size=self.batch_size,
                            shuffle=True)

    def test_dataloader(self):
        return data.DataLoader(dataset=self.test_stl10,
                                  batch_size=self.batch_size, 
                                  shuffle=False)
    
    def val_dataloader(self):
        return data.DataLoader(dataset=self.valid_stl10, 
                                  batch_size=self.batch_size,
                                  shuffle=False
                                 )
