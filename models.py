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
        self.num_blocks = options['num_blocks']
        self.block_size = options['block_size']
        self.num_classes = options['num_classes']

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
        self.blocks = nn.Sequential(nn.Linear(pretrained_model.classifier.in_features,
                                              self.num_blocks*self.block_size),
                                    nn.ReLU())
        #self.classifier = pretrained_model.classifier
        self.classifier = nn.Linear(self.num_blocks*self.block_size, self.num_classes)

    def forward(self, img, options):
        #TODO
        # Might want to impose locality on some other feature map or after ReLU
        feat_map = self.features(img) # Note that this is before the ReLU
        f = F.relu(feat_map,inplace=True)
        #f = self.avg_pool(f).view(f.size(0),-1)
        lin_feat = F.avg_pool2d(f, kernel_size=7, stride=1).view(f.size(0),-1)  # First 1-D feature
        block_logits = self.blocks(lin_feat)  # bs x (MK)
        block_splits = torch.chunk(block_logits,self.num_blocks,dim=1)  # Length -> M
                                                                        # block_splits[i] -> bs x K
        K = self.block_size
        block_sm = torch.empty_like(block_logits)  # Instead of making a list, make a tensor
                                                   # block_sm -> bs x (MK)
        if options['mode'] == 'train':
            # softmax
            for i in range(len(block_splits)):
                block_sm[:,i*K:(i+1)*K] = F.softmax(block_splits[i],dim=1)
        else:
            # put a 1 at argmax and 0 everywhere else
            for i in range(len(block_splits)):
                max_ind = torch.argmax(block_splits[i],dim=1) # List of max indices
                                                              # Length -> bs
                argmax_tensor = torch.zeros([block_splits[i].shape[0],block_splits[i].shape[1]]) # bs x K
                for j in range(argmax_tensor.shape[0]):
                    argmax_tensor[j,max_ind[j]] = 1.0
                block_sm[:,i*K:(i+1)*K] = argmax_tensor
        logits = self.classifier(block_sm)
        return feat_map, lin_feat, block_logits, logits
