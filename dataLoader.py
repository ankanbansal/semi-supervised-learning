import os
import numpy as np    
import json
import ipdb
import socket
import random

from helperFunctions import JsonProgress

from PIL import Image
from sklearn.preprocessing import  MultiLabelBinarizer, OneHotEncoder

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Data loaders

class TrainTransform(object):
    def __init__(self):
        super(TrainTransform, self).__init__()
    def __call__(self, sample):
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                         std=[0.229,0.224,0.225])
        # Might not want to do a random crop or horizontal flip
        image_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
        sample['image'] = image_transform(sample['image'])
        return sample

class ValTransform(object):
    def __init__(self):
        super(ValTransform, self).__init__()
    def __call__(self,sample):
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                         std=[0.229,0.224,0.225])
        # Might not want to do a random crop or horizontal flip
        image_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])
        sample['image'] = image_transform(sample['image'])
        return sample

class ImgDataset(Dataset):
    def __init__(self,json_file,options,transform=None, validation=False):
        print "Loading data file..."
        #self.all_files = json.load(open(json_file), object_hook=JsonProgress)
        self.all_files = json.load(open(json_file))
        self.transform = transform
        if validation:
            self.img_dir = options['val_img_dir']
        else:
            self.img_dir = options['train_img_dir']
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self,idx):
        sample = self.all_files[idx].copy()
        try:
            sample['image'] = Image.open(os.path.join(self.img_dir,sample['image_name'])).convert('RGB')
        except:
            print "Cannot load image: ", os.path.join(self.img_dir,sample['image_name'])
            return None   # This is not a good idea. Data loader will crash if some images are
                          # absent. Need to make it robust to that case. 
        if self.transform:
            sample = self.transform(sample)
            return sample

def loaders(options):
    train_file = options['train_json_file']
    val_file = options['val_json_file']

    train_transform = TrainTransform()
    val_transform = ValTransform()

    print "Creating train dataset..."
    train_dataset = ImgDataset(train_file, options, transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=options['batch_size'],
                              shuffle=True,
                              num_workers=options['num_workers'])

    if options['val_on'] or options['mode'] in ['validate','test']:
        print "Creating validation dataset..."
        val_dataset = ImgDataset(val_file, options, transform=val_transform, validation=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=options['val_batch_size'],
                                num_workers=options['num_workers'])
    else:
        val_dataset = None
        val_loader = None
    return train_loader, val_loader



class WeightedBatchSampler(object):
    def __init__(self, all_indices, sup_indices, unsup_indices, options):
        self.all_indices = all_indices
        self.batch_size = options['batch_size']
        self.sup_indices = sup_indices
        self.unsup_indices = unsup_indices
        self.ratio = options['sup_to_total_ratio']
    def __len__(self):
        return len(self.all_indices) // self.batch_size
    def __iter__(self):
        num_batches = len(self.all_indices) // self.batch_size
        while num_batches > 0:
            batch = []
            while len(batch) < self.batch_size:
                if random.random() < self.ratio:
                    batch.append(self.sup_indices[random.randint(0,len(self.sup_indices)-1)])
                else:
                    batch.append(self.unsup_indices[random.randint(0,len(self.unsup_indices)-1)])
            if len(batch) == self.batch_size:
                yield batch
            num_batches -= 1


class WeightedImgDataset(Dataset):
    def __init__(self, all_files, options, transform=None):
        print "Loading data file..."
        self.all_files = all_files
        self.transform = transform
        self.img_dir = options['train_img_dir']
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self,idx):
        sample = self.all_files[idx].copy()
        try:
            sample['image'] = Image.open(os.path.join(self.img_dir,sample['image_name'])).convert('RGB')
        except:
            print "Cannot load image: ", os.path.join(self.img_dir,sample['image_name'])
            return None   # This is not a good idea. Data loader will crash if some images are
                          # absent. Need to make it robust to that case. 
        if self.transform:
            sample = self.transform(sample)
            return sample

def weighted_loaders(options):
    sup_file_name = options['sup_json_file']
    sup_files = json.load(open(sup_file_name))
    sup_indices = range(len(sup_files))
    unsup_file_name = options['unsup_json_file']
    unsup_files = json.load(open(unsup_file_name))
    all_files = sup_files + unsup_files
    unsup_indices = range(len(sup_files),len(all_files))

    train_transform = TrainTransform()

    print 'Creating train dataset...'
    train_dataset = WeightedImgDataset(all_files, options, transform=train_transform)
    sampler = WeightedBatchSampler(range(len(all_files)), sup_indices, unsup_indices, options)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=sampler,
                              num_workers=options['num_workers'])

    val_file = options['val_json_file']
    val_transform = ValTransform()
    if options['val_on'] or options['mode'] in ['validate','test']:
        print "Creating validation dataset..."
        val_dataset = ImgDataset(val_file, options, transform=val_transform, validation=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=options['val_batch_size'],
                                num_workers=options['num_workers'])
    else:
        val_dataset = None
        val_loader = None

    return train_loader, val_loader
