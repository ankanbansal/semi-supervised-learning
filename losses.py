import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Input: block_feats -> T x (M*K)  # Where M is the number of blocks and K is the
        number of nodes per block. T is the batch size
        Output: L = MEL + BEL
        """
        #TODO
        # Make these more amenable to multi-GPU training

        #Mean Entropy Loss - For sparsity
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
        return F.pairwise_distance(group_vec, zeros, 2) # L2-norm # Do we have to specify dim?
    def forward(self, feat_map):
        """
        Input: feat_map -> T x (HxWxD)  # Where H is the height of the feature map, W is the width,
        and D is the depth. T is the batch size.
        Output: L = Locality Loss -> penalises activations with large spreads in the feature map
        """
        # Create 4 or 6 groups. And add the loss over each group.
        # Basically loop over groups and call the group_activity function. Store everything in an
        # array and return the L1 norm of the array. I think.
        #TODO
        # Find the paper and see the exact implementation
        # L1 -> top to bottom
        # L2 -> bottom to top
        # L3 -> left to right
        # L4 -> right to left

        #squared_feat_map = torch.mul(feat_map,feat_map)
        group_activity_1 = torch.zeros([feat_map.shape[0],feat_map.shape[2]]).cuda()
        #TODO
        # Try to make this faster and more amenable to multi-GPU training.
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

        tot_loss = (L1.cuda() + L2.cuda() + L3.cuda() + L4.cuda())/4.0
        return tot_loss


class LocalityEntropyLoss(torch.nn.Module):
    """
    Enforces small activation regions
    The idea is to follow a similar approach as MEL and BEL. The activation in 1 image should be
    small (hence, lower entropy). However, averaged over a mini-batch, the mean activation should be
    spread out. (And hence, higher entropy)
    """
    def __init__(self):
        super(LocalityEntropyLoss, self).__init__()
    def entropy(self, logits):
        return -1.0*(F.softmax(logits,dim=0)*F.log_softmax(logits,dim=0)).sum()
    def forward(self, feat_map):
        """
        Input: feat_map -> T x (HxWxD)  # Where H is the height of the feature map, W is the width,
        and D is the depth. T is the batch size.
        Output: L = Locality Entropy Loss MEL, and Locality Entropy Loss BEL
        """
        linearized_activations = feat_map.view(feat_map.size(0),-1)  # bs x (H*W*D)
        sum1 = torch.zeros([linearized_activations.shape[0],1])
        for t in range(linearized_activations.shape[0]):
            sum1[t] = self.entropy(linearized_activations[t,:])
        L1 = torch.mean(sum1)

        average_linealized_activation = torch.mean(linearized_activations,dim=0)
        L2 = -1.0*self.entropy(average_linealized_activation)
        
        return L1.cuda(), L2.cuda()


def get_loss(loss_name ='CE'):
    if loss_name == 'CE':
        # ignore_index ignores the samples which have label -1000. This is useful for few-shot
        # things.
        criterion = nn.CrossEntropyLoss(ignore_index=-1000).cuda()
    elif loss_name == 'ClusterLoss':
        criterion = ClusterLoss().cuda()
    elif loss_name == 'LocalityLoss':
        criterion = LocalityLoss().cuda()
    elif loss_name == 'LEL':
        criterion = LocalityEntropyLoss().cuda()

    return criterion


# This loss contained loss in the case of block softmax. Not using this.
# Keeping it here for posterity
#class IncorrectClusterLoss(torch.nn.Module):
#    """
#    Cluster loss is comes from the SuBiC paper and consists of two losses.
#    First is the Mean Entropy Loss which makes the block features to be close to one-hot encoded
#    vectors.
#    Second is the Batch Entropy Loss which ensures a uniform distribution o activations over the
#    block features (Uniform block support). 
#    """
#    def __init__(self,options):
#        super(ClusterLoss, self).__init__()
#        self.num_blocks = options['num_blocks']
#        self.lmbda = options['lmbda']
#    def entropy(self, prob_dist):
#        return -1.0*(F.softmax(prob_dist,dim=0)*F.log_softmax(prob_dist,dim=0)).sum()
#    def forward(self, block_feats):
#        """
#        Input: block_feats -> T x (M*K)  # Where M is the number of blocks and K is the
#        number of nodes per block. T is the batch size
#        Output: L = MEL + BEL
#        """
#        M = self.num_blocks
#        #Mean Entropy Loss - For sparsity
#        #  L1 = Sum_batch_i(Sum_block_m(Entropy(block_i_m)))/TM
#        blocks = torch.chunk(block_feats, M, dim=1)  # blocks contains M chunks. Each with shape TxK
#
#        sum1 = torch.zeros([len(blocks),1])
#        for m in range(len(blocks)):
#            entropies = torch.zeros([blocks[m].shape[0],1])
#            for t in range(blocks[m].shape[0]):
#                entropies[t] = self.entropy(blocks[m][t,:])
#            sum1[m] = torch.mean(entropies)
#        L1 = torch.mean(sum1)
#
#        # Alternate implementation of MEL - Is this wrong?
#        #sum1 = torch.zeros([block_feats.shape[0],1])
#        #for t in range(block_feats.shape[0]):
#        #    image_sum = None
#        #    for m in range(M):
#        #        if image_sum is None:
#        #            image_sum = self.entropy(blocks[m][t,:])
#        #        else:
#        #            image_sum += self.entropy(blocks[m][t,:])
#        #    sum1[t] = image_sum/M
#        #L1 = torch.mean(sum1)
#        
#        #Batch Entropy Loss - For uniform support
#        #  L2 = -Sum_block_m(Entropy(Sum_batch_i(block_i_m)/T))/M
#        sum2 = None
#        for m in range(M):
#            block_mean = torch.mean(blocks[m],dim=0)
#            if sum2 is None:
#                sum2 = self.entropy(block_mean)
#            else:
#                sum2 += self.entropy(block_mean)
#        L2 = -1.0*sum2/M
#        #L2 = torch.Tensor([0.0])
#
#        L = L1.cuda() + self.lmbda*L2.cuda()   # These losses probably aren't being computed on GPU
#        #L = self.lmbda*L2.cuda()   #lambda should be positive because there is already a -1.0 in L2
#        #TODO
#        # 3. Use torch operations instead of for loops if you want to use multiple GPUs!
#        return L1, L2, L
