import numpy as np
import ipdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.models as tv_models
from torchvision import transforms

from densenet import densenet_cifar


#class WSODModel(nn.Module):
#    def __init__(self, options):
#        super(WSODModel, self).__init__()
#        
#        self.arch = options['base_arch']
#        
#        if self.arch == 'densenet121':
#            pretrained_model = tv_models.densenet121(pretrained=False,growth_rate=12)
#            self.features = pretrained_model.features
#            #self.classifier = pretrained_model.classifier
#            self.classifier = nn.Linear(pretrained_model.classifier.in_features,options['num_classes'])
#        elif self.arch == 'resnet18':
#            pretrained_model = tv_models.resnet18(pretrained=False)
#            self.conv1 = pretrained_model.conv1
#            self.bn1 = pretrained_model.bn1
#            self.relu = pretrained_model.relu
#            self.maxpool = pretrained_model.maxpool
#            self.layer1 = pretrained_model.layer1
#            self.layer2 = pretrained_model.layer2
#            self.layer3 = pretrained_model.layer3
#            self.layer4 = pretrained_model.layer4
#            #self.avgpool = pretrained_model.avgpool
#            #self.classifier = pretrained_model.fc
#            self.classifier = nn.Linear(pretrained_model.fc.in_features,options['num_classes'])
#
#    def forward(self, img, options):
#        if self.arch == 'resnet18':
#            c1 = self.conv1(img)
#            bn1 = self.bn1(c1)
#            relu1 = self.relu(bn1)
#            mp1 = self.maxpool(relu1)
#            l1 = self.layer1(mp1)
#            l2 = self.layer2(l1)
#            l3 = self.layer3(l2)
#            feat_map_relu = self.layer4(l3)
#            #lin_feat = self.avgpool(feat_map_relu)
#            #lin_feat = lin_feat.view(lin_feat.size(0),-1)
#            lin_feat = feat_map_relu.view(feat_map_relu.size(0),-1)
#            #lin_feat = self.avgpool(feat_map_relu).view(feat_map_relu.size(0),-1)  # First 1-D feature
#        else:
#            feat_map = self.features(img) 
#            feat_map_relu = F.relu(feat_map,inplace=True)
#            #lin_feat = F.avg_pool2d(feat_map_relu, kernel_size=7, stride=1).view(feat_map_relu.size(0),-1)  # First 1-D feature
#            lin_feat = feat_map_relu.view(feat_map_relu.size(0),-1)
#
#        logits = self.classifier(lin_feat)
#
#        final_output = F.softmax(logits)
#        #return feat_map, logits, final_output
#        return feat_map_relu, logits, final_output
#
#
class WSODModel(nn.Module):
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
            logits = self.pretrained_model(img)
        elif self.arch == 'densenet121':
            feat_map = self.features(img) 
            feat_map_relu = F.relu(feat_map,inplace=True)
            lin_feat = feat_map_relu.view(feat_map_relu.size(0),-1)
            logits = self.classifier(lin_feat)

        final_output = F.softmax(logits)
        return logits, final_output
