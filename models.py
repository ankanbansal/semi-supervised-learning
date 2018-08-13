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
            pretrained_model = tv_models.densenet161(pretrained=False)
        elif arch == 'densenet169':
            pretrained_model = tv_models.densenet169(pretrained=False)
        elif arch == 'densenet201':
            pretrained_model = tv_models.densenet201(pretrained=False)
        elif arch == 'resnet152':
            pretrained_model = tv_models.resnet152(pretrained=False)

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
        super(WSODModel, self).__init__()

        self.arch = options['base_arch']
        if self.arch == 'densenet161':
            pretrained_model = tv_models.densenet161(pretrained=False)
            self.features = pretrained_model.features
            self.classifier = pretrained_model.classifier
        elif self.arch == 'densenet169':
            pretrained_model = tv_models.densenet169(pretrained=False)
            self.features = pretrained_model.features
            self.classifier = pretrained_model.classifier
        elif self.arch == 'densenet201':
            pretrained_model = tv_models.densenet201(pretrained=False)
            self.features = pretrained_model.features
            self.classifier = pretrained_model.classifier
        elif self.arch == 'resnet152':
            pretrained_model = tv_models.resnet152(pretrained=False)
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.relu = pretrained_model.relu
            self.maxpool = pretrained_model.maxpool
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
            self.avgpool = pretrained_model.avgpool
            self.classifier = pretrained_model.fc

    def forward(self, img, options):
        #TODO
        # Might want to impose locality on some other feature map
        if self.arch == 'resnet152':
            c1 = self.conv1(img)
            bn1 = self.bn1(c1)
            relu1 = self.relu(bn1)
            mp1 = self.maxpool(relu1)
            l1 = self.layer1(mp1)
            l2 = self.layer2(l1)
            l3 = self.layer3(l2)
            feat_map_relu = self.layer4(l3)
            lin_feat = self.avgpool(feat_map_relu)
            lin_feat = lin_feat.view(lin_feat.size(0),-1)
            #lin_feat = self.avgpool(feat_map_relu).view(feat_map_relu.size(0),-1)  # First 1-D feature
        else:
            feat_map = self.features(img) 
            feat_map_relu = F.relu(feat_map,inplace=True)
            lin_feat = F.avg_pool2d(feat_map_relu, kernel_size=7, stride=1).view(feat_map_relu.size(0),-1)  # First 1-D feature

        logits = self.classifier(lin_feat)

        final_output = F.softmax(logits)
        return feat_map_relu, logits, final_output


class WSODModel_LargerCAM(nn.Module):
    def __init__(self,options):
        super(WSODModel_LargerCAM, self).__init__()

        pretrained_model = tv_models.densenet161(pretrained=False)

        self.feature_map = nn.Sequential(pretrained_model.features.conv0, pretrained_model.features.norm0,
                                      pretrained_model.features.relu0, pretrained_model.features.pool0,
                                      pretrained_model.features.denseblock1, pretrained_model.features.transition1,
                                      pretrained_model.features.denseblock2, pretrained_model.features.transition2, 
                                      pretrained_model.features.denseblock3, pretrained_model.features.transition3.norm,
                                      pretrained_model.features.transition3.relu, pretrained_model.features.transition3.conv)
        self.norm = nn.BatchNorm2d(1056)
        self.classifier1 = nn.Linear(in_features=1056,out_features=1000)

        self.small_cnn = nn.Sequential(pretrained_model.features.transition3.pool, pretrained_model.features.denseblock4, 
                                       pretrained_model.features.norm5)
        self.classifier = pretrained_model.classifier

    def forward(self, img, options):
        feat_map = self.feature_map(img)
        norm1 = self.norm(feat_map)
        feat_map_relu = F.relu(feat_map,inplace=True)
        lin_feat1 = F.avg_pool2d(feat_map_relu, kernel_size=14, stride=1).view(feat_map_relu.size(0),-1)
        logits1 = self.classifier1(lin_feat1)
        final_output1 = F.softmax(logits1)

        feat_map_small = self.small_cnn(feat_map)
        feat_map_relu_small = F.relu(feat_map_small,inplace=True)
        lin_feat2 = F.avg_pool2d(feat_map_relu_small, kernel_size=7, stride=1).view(feat_map_relu_small.size(0),-1)  # First 1-D feature
        logits2 = self.classifier(lin_feat2)
        
        return feat_map_relu, logits1, logits2, final_output1
