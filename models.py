import numpy as np
import ipdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.models as tv_models
from torchvision import transforms

# Model definitions

class BasicClassificationModel(nn.Module):
    def __init__(self,options):
        super(BasicClassificationModel, self).__init__()
        arch = options['base_arch']
        if arch == 'densenet161':
            pretrained_model = tv_models.densenet161(pretrained=True)
        elif arch == 'densenet169':
            pretrained_model = tv_models.densenet169(pretrained=True)
        elif arch == 'densenet201':
            pretrained_model = tv_models.densenet201(pretrained=True)
        elif arch == 'resnet152':
            pretrained_model = tv_models.resnet152(pretrained=True)

        self.features = pretrained_model.features
        #self.avg_pool = nn.AvgPool2d(7)
        self.classifier = pretrained_model.classifier

    def forward(self, img):
        f = self.features(img)
        f = F.relu(f, inplace=True)
        #f = self.avg_pool(f).view(f.size(0),-1)
        f = F.avg_pool2d(f, kernel_size=7, stride=1).view(f.size(0), -1)
        y = self.classifier(f)
        return y


class WSODModel(nn.Module):
    def __init__(self,options):
        super(FirstModel, self).__init__()
        arch = options['base_arch']
        if arch == 'densenet161':
            pretrained_model = tv_models.densenet161(pretrained=True)
        elif arch == 'densenet169':
            pretrained_model = tv_models.densenet169(pretrained=True)
        elif arch == 'densenet201':
            pretrained_model = tv_models.densenet201(pretrained=True)
        elif arch == 'resnet152':
            pretrained_model = tv_models.resnet152(pretrained=True)

        self.features = pretrained_model.features
        #self.avg_pool = nn.AvgPool2d(7)
        self.blocks = nn.Sequential(nn.Linear(pretrained_model.feature.out_features, _____),
                                    nn.ReLU())
        self.argmax = ____
        #self.classifier = pretrained_model.classifier
        self.classifier = nn.Linear(_____)

    def forward(self, img, M, options):
        feat_map = self.features(img) # Note that this is before the ReLU
        f = F.relu(f,inplace=True)
        #f = self.avg_pool(f).view(f.size(0),-1)
        lin_feat = F.avg_pool2d(f, kernel_size=7, stride=1).view(f.size(0),-1)  # First 1-D feature
        block_logits = self.blocks(lin_feat)
        # TODO
        #1. Number of chunks - M?
        block_splits = torch.chunk(block_logits,M,dim=1)
        K = block_logits.shape[-1]/M
        block_sm = torch.empty_like(block_logits)  # Instead of making a list, make a tensor
        if options['mode'] == 'train':
            # softmax
            for i in range(len(block_splits)):
                block_sm[:,i*K:(i+1)*K] = F.softmax(block_splits[i])
        else:
            # put a 1 at argmax and 0 everywhere else
            for i in range(len(block_splits)):
                block_sm[:,i*K:(i+1)*K] = self.______(block_splits[i])
        logits = self.classifier(block_sm)
        return feat_map, lin_feat, block_logits, logits
