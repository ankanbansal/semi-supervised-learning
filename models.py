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

        if options['mode'] == 'train':
            # softmax
            final_output = F.softmax(logits)
        else:
            # put a 1 at argmax and 0 everywhere else
            #max_ind = torch.argmax(logits,dim=1)
            #final_output = torch.zeros([logits.shape[0],logits.shape[1]]) # bs x C
            #for j in range(final_output.shape[0]):
            #    final_output[j,max_ind[j]] = 1.0
            final_output = F.softmax(logits)
        #return feat_map, logits, final_output
        return feat_map_relu, logits, final_output
