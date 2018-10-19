import numpy as np
# import ipdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.models as tv_models
from torchvision import transforms

from densenet import densenet_cifar

class WSODModel(nn.Module):
    """
        The best performance is achieved by densenet_cifar model. Implementation has been obtained
        from: https://github.com/kuangliu/pytorch-cifar'''
    """
    def __init__(self, options):
        super(WSODModel, self).__init__()
        self.arch = options['base_arch']
        if self.arch == 'densenet_cifar':
            self.pretrained_model = densenet_cifar()
        elif self.arch == 'densenet121':
            pretrained_model = tv_models.densenet121(pretrained=False,growth_rate=12)
            self.features = pretrained_model.features
            self.classifier = nn.Linear(pretrained_model.classifier.in_features,options['num_classes'])

    def forward(self, img, options):
        if self.arch == 'densenet_cifar':
            lin_feat, logits = self.pretrained_model(img)
        elif self.arch == 'densenet121':
            feat_map = self.features(img)
            feat_map_relu = F.relu(feat_map,inplace=True)
            lin_feat = feat_map_relu.view(feat_map_relu.size(0),-1)
            logits = self.classifier(lin_feat)

        final_output = F.softmax(logits)
        return lin_feat, logits, final_output

