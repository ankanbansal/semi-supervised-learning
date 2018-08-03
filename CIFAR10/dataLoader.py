import numpy as np
import random
import ipdb

import torch
import torchvision as tv
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader


class TrainTransform(object):
    def __init__(self):
        super(TrainTransform, self).__init__()
    def __call__(self, sample):
        normalize = tv_transforms.Normalize(mean=[0.5,0.5,0.5],
                                         std=[0.5,0.5,0.5])
        image_transform = tv_transforms.Compose([tv_transforms.RandomHorizontalFlip(),
                                              tv_transforms.ToTensor(),
                                               normalize])
        sample = image_transform(sample)
        return sample


class ValTransform(object):
    def __init__(self):
        super(ValTransform, self).__init__()
    def __call__(self, sample):
        normalize = tv_transforms.Normalize(mean=[0.5,0.5,0.5],
                                         std=[0.5,0.5,0.5])
        image_transform = tv_transforms.Compose([tv_transforms.ToTensor(),
                                                 normalize])
        sample = image_transform(sample)
        return sample


class WeightedBatchSampler(object):
    def __init__(self, all_indices, sup_indices, unsup_indices, options):
        self.all_indices = all_indices
        self.batch_size = options['batch_size']
        self.sup_indices = sup_indices
        self.unsup_indices = unsup_indices
        self.ratio = options['sup_to_tot_ratio']
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


def weighted_loaders(options):
    all_indices = list(range(options['dataset_size']))
    sup_indices = np.random.choice(all_indices, size=options['num_sup'], replace=False)
    unsup_indices = list(set(all_indices) - set(sup_indices))

    train_transform = TrainTransform()
    train_dataset = tv.datasets.CIFAR10(root=options['data_dir'], train=True, transform=train_transform)
    
    for i in range(len(train_dataset.train_labels)):
        if i in sup_indices:
            continue
        else:
            train_dataset.train_labels[i] = -1000

    sampler = WeightedBatchSampler(all_indices, sup_indices, unsup_indices, options)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=sampler,
                              num_workers=options['num_workers'])

    val_transform = ValTransform()
    if options['val_on'] or options['mode'] in ['validate', 'test']:
        print "Creating validation dataset..."
        val_dataset = tv.datasets.CIFAR10(root=options['data_dir'], train=False,
                transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=options['val_batch_size'],
                                shuffle=False, num_workers=options['num_workers'])
    else:
        val_dataset = None
        val_loader = None

    return train_loader, val_loader
