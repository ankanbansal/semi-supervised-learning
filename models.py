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
        self.avg_pool = nn.AvgPool2d(7)
        self.classifier = pretrained_model.classifier

    def forward(self, img):
        f = self.features(img)
        f = F.relu(f, inplace=True)
        f = self.avg_pool(f).view(f.size(0),-1)
        y = self.classifier(f)
        # Note that this is not doing softmax. Loss takes care of that. During test/validation have
        # to explicitly do a softmax
        return y


# Model definitions
class FirstModel(nn.Module):
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
        self.avg_pool = nn.AvgPool2d(7)
        in_feats = pretrained_model.classifier.in_features + .....
        out_feats = .....
        self.classifier = (nn.Linear(out_feats, options['num_classes']))

    def forward(self, img):
