import numpy as np
import random
import ipdb

import torch
import torchvision as tv
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader


# Transforms taken from near state-of-the-art densenet model
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# The best Densenet model from the paper achieved 3.46% while our model achieves 5.13%.
# I couldn't find an implementation that achieves 3.46%. So I am using the one which gets 5.13%.
class TrainTransform(object):
    def __init__(self):
        super(TrainTransform, self).__init__()
    def __call__(self, sample):
        normalize = tv_transforms.Normalize(mean=[0.49139968,0.48215827,0.44653124],
                                         std=[0.2023,0.1994,0.2010])
        image_transform = tv_transforms.Compose([tv_transforms.RandomCrop(32,padding=4),
                                                 tv_transforms.RandomHorizontalFlip(),
                                                 tv_transforms.ToTensor(),
                                                 normalize])
        sample = image_transform(sample)
        return sample

class ValTransform(object):
    def __init__(self):
        super(ValTransform, self).__init__()
    def __call__(self, sample):
        normalize = tv_transforms.Normalize(mean=[0.49139968,0.48215827,0.44653124],
                                         std=[0.2023,0.1994,0.2010])
        image_transform = tv_transforms.Compose([tv_transforms.ToTensor(),
                                                 normalize])
        sample = image_transform(sample)
        return sample

# "Weighted" refers to weighing of the ratio of supervised and unsupervised samples
class WeightedBatchSampler(object):
    def __init__(self, dataset_size, num_sup, num_unsup, batch_size, sup_to_tot_ratio):
        self.all_indices = list(range(dataset_size))
        self.sup_indices = np.random.choice(self.all_indices, size=num_sup, replace=False)
        self.unsup_indices = list(set(self.all_indices) - set(self.sup_indices))[:num_unsup]
        self.batch_size = batch_size
        self.ratio = sup_to_tot_ratio
    def __len__(self):
        return (len(self.sup_indices) + len(self.unsup_indices)) // self.batch_size
    def __iter__(self):
        num_batches = (len(self.sup_indices) + len(self.unsup_indices)) // self.batch_size
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
    train_transform = TrainTransform()
    train_dataset = tv.datasets.CIFAR10(root=options['data_dir'], train=True, transform=train_transform)

    # Each batch contains the some supervised images and some unsupervised images depending on
    # sup_to_tot_ratio
    sampler = WeightedBatchSampler(train_dataset.train_data.shape[0], options['num_sup'],
            options['num_unsup'], options['batch_size'], options['sup_to_tot_ratio'])

    for i in sampler.unsup_indices:
        train_dataset.train_labels[i] = -1000

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
