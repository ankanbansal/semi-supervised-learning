import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb
import time

# Clustering penalties
class ClusterLoss(torch.nn.Module):
    """
    Cluster loss comes from the SuBiC paper and consists of two losses. First is the Mean Entropy
    Loss which makes the output to be close to one-hot encoded vectors. 
    Second is the Negative Batch Entropy Loss which ensures a uniform distribution of activations
    over the output (Uniform block support).
    """
    def __init__(self):
        super(ClusterLoss, self).__init__()
    def entropy(self, logits):
        return -1.0*(F.softmax(logits,dim=0)*F.log_softmax(logits,dim=0)).sum()
    def forward(self, logits):
        """
        Input: logits -> T x K # Where K is the number of classes and T is the batch size
        Output: L = MEL, BEL
        """
        # Mean Entropy Loss - For one-hotness
        #   L1 = Sum_batch_i(Sum_block_m(Entropy(block_i_m)))/TM
        sum1 = torch.zeros([logits.shape[0],1])
        for t in range(logits.shape[0]):
            sum1[t] = self.entropy(logits[t,:])
        L1 = torch.mean(sum1)

        # Batch Entropy Loss - For uniform support
        #   L2 = -Sum_block_m(Entropy(Sum_batch_i(block_i_m)/T))/M
        mean_output = torch.mean(logits, dim=0)
        L2 = -1.0*self.entropy(mean_output)

        return L1.cuda(), L2.cuda()


# Stochastic Transformation Stability Loss. Introduced in:
# "Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised
# Learning"
class StochasticTransformationLoss(torch.nn.Module):
    """
    The idea behind this is that stochastic transformations of an image (flips and translations)
    should lead to very close features
    """
    def __init__(self):
        super(StochasticTransformationLoss, self).__init__()
    def entropy(self, logits):
        """
        Input: logits -> N x 1 x D # Where D is the feature dimension
        Output: entropy -> N x 1
        """
        # TODO
        # Check is this is correct
        return -1.0*(F.softmax(logits,dim=-1)*F.log_softmax(logits,dim=-1)).sum(-1)
    def cross_entropy(self, logits1, logits2):
        """
        Input: logits1 -> N x 1 x D # Where D is the feature dimension
               logits2 -> 1 x N x D # Where D is the feature dimension
        Output: Pairwise Cross-entropy -> N x N
        """
        # TODO
        # Check is this is correct
        return -1.0*(F.softmax(logits1,dim=-1)*F.log_softmax(logits2,dim=-1)).sum(-1)
    def distances(self, A, distance_type='Euclidean', eps=1e-6):
        """
        Input: A -> num_transformations x D # Where D is the feature dimension
               distance_type -> 'Euclidean'/'cosine'/'KL'
        Output: distances -> num_transformations x num_transformations pair wise distances
        """
        assert A.dim() == 2
        if distance_type == 'Euclidean':
            # 1. Numerically stable but too much memory?
            B = A.unsqueeze(1)
            C = A.unsqueeze(0)
            differences = B - C
            distances = torch.sum(differences*differences,-1) # N x N
            # Do we need sqrt? - Paper doesn't do sqrt
            # 2. Less memory but numerically unstable due to rounding errors
            #A_norm_1 = (A**2).sum(1).view(-1,1)
            #A_norm_2 = A_norm_1.view(1,-1)
            #distances = A_norm_1 + A_norm_2 - 2.0*torch.matmul(A, torch.transpose(A,0,1))
        elif distance_type == 'cosine':
            B = F.normalize(A, p=2, dim=1)
            distances = 1.0 - torch.matmul(B,B.t()) # N x N
        elif distance_type == 'KL':
            # Make sure that A contains logits
            B = A.unsqueeze(1) 
            C = A.unsqueeze(0)
            # TODO
            # Might have to use a symmetric KL div
            # Check - Still probably incorrect. Probably due to incorrect cross_entropy
            # implementation
            distances = -1.0*self.entropy(B) + self.cross_entropy(B,C) # N x N
        return distances
    def forward(self, features, num_transformations, distance_type='Euclidean'):
        """
        Input: features -> T x D # Where D is the feature dimension and T is the batch size
               num_transformations -> Number of transformations applied to the data
               (Make sure that T is a multiple of num_transformations)
        Output: ST Loss
        """
        batch_size = features.shape[0]
        #split_features = torch.zeros([batch_size/num_transformations, num_transformations, features.shape[1]])
        all_index_groups = [[(i*num_transformations)+j for j in range(num_transformations)] for i in range(batch_size/num_transformations)]

        total_loss = 0.0

        for i in range(len(all_index_groups)):
            split_features = torch.index_select(features, 0, torch.cuda.LongTensor(all_index_groups[i]))
            distances = self.distances(split_features,distance_type=distance_type)
            total_loss += 0.5*torch.sum(distances)

        total_loss = total_loss / (1.0*batch_size)

        # Don't know how exactly should we average. Per pair? Per image?
        return total_loss


def get_loss(loss_name = 'CE'):
    if loss_name == 'CE':
        # ignore_index ignores the samples which have label -1000. We specify the unsupervised images by
        # label 1000
        criterion = nn.CrossEntropyLoss(ignore_index = -1000).cuda()
    elif loss_name == 'ClusterLoss':
        criterion = ClusterLoss().cuda()
    elif loss_name == 'LocalityLoss':
        criterion = LocalityLoss().cuda()
    elif loss_name == 'CAMLocalityLoss':
        criterion = CAMLocalityLoss().cuda()
    elif loss_name == 'LEL':
        criterion = LocalityEntropyLoss().cuda()
    elif loss_name == 'STLoss':
        criterion = StochasticTransformationLoss().cuda()
    
    return criterion
