import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom loss definitions
class ClusterLoss(torch.nn.Module):
    """
    Cluster loss is comes from the SuBiC paper and consists of two losses.
    First is the Mean Entropy Loss which makes the block features to be close to one-hot encoded
    vectors.
    Second is the Batch Entropy Loss which ensures a uniform distribution o activations over the
    block features (Uniform block support). 
    """
    def __init__(self,options):
        super(ClusterLoss, self).__init__()
        self.num_blocks = options['num_blocks']
        self.lmbda = options['lmbda']
    def entropy(self, prob_dist):
        return -1.0*(F.softmax(prob_dist,dim=0)*F.log_softmax(prob_dist,dim=0)).sum()
    def forward(self, block_feats):
        """
        Input: block_feats -> T x (M*K)  # Where M is the number of blocks and K is the
        number of nodes per block. T is the batch size
        Output: L = MEL + BEL
        """
        M = self.num_blocks
        #Mean Entropy Loss - For sparsity
        #  L1 = Sum_batch_i(Sum_block_m(Entropy(block_i_m)))/TM
        blocks = torch.chunk(block_feats, M, dim=1)  # blocks contains M chunks. Each with shape TxK
        sum1 = torch.zeros([block_feats.shape[0],1])
        for i in range(block_feats.shape[0]):
            image_sum = None
            for m in range(M):
                if image_sum is None:
                    image_sum = self.entropy(blocks[m][i,:])
                else:
                    image_sum += self.entropy(blocks[m][i,:])
            sum1[i] = image_sum/M
        # This might not work because sum1 is a list
        L1 = torch.mean(sum1)
        
        #Batch Entropy Loss - For uniform support
        #  L2 = -Sum_block_m(Entropy(Sum_batch_i(block_i_m)/T))/M
        sum2 = None
        for m in range(M):
            block_mean = torch.mean(blocks[m],dim=0)
            if sum2 is None:
                sum2 = self.entropy(block_mean)
            else:
                sum2 += self.entropy(block_mean)
        L2 = -1.0*sum2/M

        L = L1.cuda() + self.lmbda*L2.cuda()
        #TODO
        # 1. Define lmbda
        # 3. Use torch operations instead of for loops if you want to use multiple GPUs ???
        return L


class LocalityLoss(torch.nn.Module):
    """
    Enforces small activation regions
    """
    def __init__(self):
        super(LocalityLoss, self).__init__()
    def group_activity(self,group):
        # Function to calculate the group activity
        # Basically just vectorizes the whole group and calculates the L2 norm
        group_vec = group.contiguous().view(group.size(0),-1)  # The size -1 is inferred from the other
        # dimensions. This essentially keeps the batch_size same and vectorizes everything else
        #group_vec is now bs x num_pixels_in_group
        zeros = torch.empty_like(group_vec)
        return F.pairwise_distance(group_vec, zeros, 2) # L2-norm
    def forward(self, feat_map):
        """
        Input: feat_map -> T x (HxWxD)  # Where H is the height of the feature map, W is the width,
        and D is the depth. T is the batch size.
        Output: L = Locality Loss -> penalises activations with large spreads in the feature map
        """
        # Create 4 or 6 groups. And add the loss over each group.
        # Basically loop over groups and call the group_activity function. Store everything in an
        # array and return the L1 norm of the array. I think.
        # TODO
        # Find the paper and see the exact implementation
        # L1 -> top to bottom
        # L2 -> bottom to top
        # L3 -> left to right
        # L4 -> right to left
        group_activity_1 = torch.zeros([feat_map.shape[0],feat_map.shape[2]]).cuda()
        #TODO
        # Does the .cuda() help or make sense?
        for i in range(feat_map.shape[2]):
            group = feat_map[:,:,i:,:] 
            group_activity_1[:,i] = self.group_activity(group)
        zeros = torch.empty_like(group_activity_1)
        L1 = torch.mean(F.pairwise_distance(group_activity_1,zeros,1))   # L1-norm

        group_activity_2 = torch.zeros([feat_map.shape[0],feat_map.shape[2]]).cuda()
        for j in reversed(range(feat_map.shape[2])):
            group = feat_map[:,:,:j+1,:]
            group_activity_2[:,j] = self.group_activity(group)
        zeros = torch.empty_like(group_activity_2)
        L2 = torch.mean(F.pairwise_distance(group_activity_2,zeros,1)) 

        group_activity_3 = torch.zeros([feat_map.shape[0],feat_map.shape[3]]).cuda()
        for k in range(feat_map.shape[3]):
            group = feat_map[:,:,:,k:]
            group_activity_3[:,k] = self.group_activity(group)
        zeros = torch.empty_like(group_activity_3)
        L3 = torch.mean(F.pairwise_distance(group_activity_3,zeros,1)) 

        group_activity_4 = torch.zeros([feat_map.shape[0],feat_map.shape[3]]).cuda()
        for l in reversed(range(feat_map.shape[3])):
            group = feat_map[:,:,:,:l+1]
            group_activity_4[:,l] = self.group_activity(group)
        zeros = torch.empty_like(group_activity_4)
        L4 = torch.mean(F.pairwise_distance(group_activity_4,zeros,1))

        tot_loss = (L1 + L2 + L3 + L4)/4.0
        return tot_loss


def get_loss(options,loss_name ='CE'):
    if loss_name == 'CE':
        criterion = nn.CrossEntropyLoss().cuda()
    elif loss_name == 'ClusterLoss':
        criterion = ClusterLoss(options).cuda()
    elif loss_name == 'LocalityLoss':
        criterion = LocalityLoss().cuda()

    return criterion
