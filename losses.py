import torch
import numpy as np

import torch
import torch.nn as nn

# Custom loss definitions
#class ClusterLoss(torch.nn.Module):
#    def __init__(self):
#        super(ClusterLoss, self).__init__()
#    def forward(self, ____, ____):
#
#
#class LocalityLoss(torch.nn.Module):
#    def __init__(self):
#        super(LocalityLoss, self).__init__()
#    def forward(self, _____, ____):


def get_loss(loss_name ='CE'):
    if loss_name == 'CE':
        criterion = nn.CrossEntropyLoss().cuda()
    elif loss_name == 'ClusterLoss':
        criterion = ClusterLoss().cuda()
    elif loss_name == 'LocalityLoss':
        criterion = LocalityLoss().cuda()

    return criterion
