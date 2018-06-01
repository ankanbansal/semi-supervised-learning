import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.funcitonal as F

# Custom loss definitions
class ClusterLoss(torch.nn.Module):
    """
    Cluster loss is comes from the SuBiC paper and consists of two losses.
    First is the Mean Entropy Loss which makes the block features to be close to one-hot encoded
    vectors.
    Second is the Batch Entropy Loss which ensures a uniform distribution o activations over the
    block features (Uniform block support). 
    """
    def __init__(self):
        super(ClusterLoss, self).__init__()
    def entropy(self, prob_dist):
        return -1.0*(F.softmax(prob_dist,dim=0)*F.log_softmax(prob_dist,dim=0)).sum()
    def forward(self, block_feats, M):
        """
        Input: block_feats -> T x (M*K)  # Where M is the number of blocks and K is the
        number of nodes per block. T is the batch size
        Output: L = MEL + BEL
        """
        #Mean Entropy Loss - For sparsity
        #  L1 = Sum_batch_i(Sum_block_m(Entropy(block_i_m)))/TM
        blocks = torch.chunk(block_feats, M, dim=1)  # blocks contains M chunks. Each with shape TxK
        sum1 = []
        for i in range(block_feats.shape[0]):
            image_sum = 0.0
            for m in range(M):
                image_sum += self.entropy(blocks[m][i,:])
            sum1.append(image_sum/M)
        # This might not work because sum1 is a list
        L1 = torch.mean(sum1)
        
        #Batch Entropy Loss - For uniform support
        #  L2 = -Sum_block_m(Entropy(Sum_batch_i(block_i_m)/T))/M
        sum2 = 0.0
        for m in range(M):
            block_mean = torch.mean(blocks[m],dim=0)
            sum2 += self.entropy(block_mean)
        L2 = -1.0*sum2/M

        L = L1 + lmbda*L2
        #TODO
        # 1. Define lmbda
        # 2. Pass M
        # 3. Use torch operations instead of for loops if you want to use multiple GPUs

        


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
