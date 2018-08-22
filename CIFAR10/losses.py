import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import ipdb
import time


# Clustering penalties
class ClusterLoss(torch.nn.Module):
    """
    Cluster loss is comes from the SuBiC paper and consists of two losses.
    First is the Mean Entropy Loss which makes the output to be close to one-hot encoded
    vectors.
    Second is the Batch Entropy Loss which ensures a uniform distribution o activations over the
    output (Uniform block support). 
    """
    def __init__(self):
        super(ClusterLoss, self).__init__()
    def entropy(self, logits):
        return -1.0*(F.softmax(logits,dim=0)*F.log_softmax(logits,dim=0)).sum()
    def forward(self, logits):
        """
        Input: block_feats -> T x K  # Where K is the number of classes and T is the batch size
        Output: L = MEL, BEL
        """
        #Mean Entropy Loss - For one-hotness
        #  L1 = Sum_batch_i(Sum_block_m(Entropy(block_i_m)))/TM
        sum1 = torch.zeros([logits.shape[0],1])
        for t in range(logits.shape[0]):
            sum1[t] = self.entropy(logits[t,:])
        L1 = torch.mean(sum1)

        #Batch Entropy Loss - For uniform support
        #  L2 = -Sum_block_m(Entropy(Sum_batch_i(block_i_m)/T))/M
        mean_output = torch.mean(logits,dim=0)
        L2 = -1.0*self.entropy(mean_output)

        return L1.cuda(), L2.cuda()


# Stochastic Transformation Stability Loss. Introduced in: 
# "Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised
# Learning"
class StochasticTransformationLoss(torch.nn.Module):
    """
    The idea behind this is that stochastic transformations of an image (flips and translations)
    should lead to very close features.
    """
    def __init__(self):
        super(StochasticTransformationLoss, self).__init__()
    def forward(self, features, num_transformations):
        """
        Input: features -> T x D # Where D is the feature dimension and T is the batch size
               num_transformations -> Number of transformations applied to the data
               Make sure that T is a multiple of num_transformations
        Output: ST Loss 
        """
        batch_size = features.shape[0]
        split_features = torch.zeros([num_transformations, batch_size/num_transformations,
            features.shape[1]])
        for i in range(num_transformations):
            indices = torch.Tensor(range(i, batch_size, num_transformations))  # Has to be a longtensor
            split_features[i,:,:] = torch.index_select(feature, 0, indices)

        



def get_loss(loss_name ='CE'):
    if loss_name == 'CE':
        # ignore_index ignores the samples which have label -1000. We specify the unsupervised images by label -1000
        criterion = nn.CrossEntropyLoss(ignore_index=-1000).cuda()
    elif loss_name == 'ClusterLoss':
        criterion = ClusterLoss().cuda()
    elif loss_name == 'LocalityLoss':
        criterion = LocalityLoss().cuda()
    elif loss_name == 'CAMLocalityLoss':
        criterion = CAMLocalityLoss().cuda()
    elif loss_name == 'LEL':
        criterion = LocalityEntropyLoss().cuda()

    return criterion
