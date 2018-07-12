import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb
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
        Input: block_feats -> T x (M*K)  # Where M is the number of blocks and K is the
        number of nodes per block. T is the batch size
        Output: L = MEL + BEL
        """
        #TODO
        # Make these more amenable to multi-GPU training

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


# Locality loss (group sparsity) over Feature Maps
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
        #zeros = torch.empty_like(group_vec)
        zeros = torch.zeros([group_vec.shape[0],group_vec.shape[1]]).cuda()
        return F.pairwise_distance(group_vec, zeros, 2) # L2-norm # Do we have to specify dim?
    def forward(self, feat_map):
        """
        Input: feat_map -> T x (DxHxW)  # Where H is the height of the feature map, W is the width,
        and D is the depth. T is the batch size.
        Output: L = Locality Loss -> penalises activations with large spreads in the feature map
        """
        # Create 4 or 6 groups. And add the loss over each group.
        # Basically loop over groups and call the group_activity function. Store everything in an
        # array and return the L1 norm of the array.
        # See the Sparse PCA paper for more details of the idea

        # L1 -> top to bottom
        # L2 -> bottom to top
        # L3 -> left to right
        # L4 -> right to left

        # This implementation is correct. Only less efficient than the later implementation.
        group_activity_1 = torch.zeros([feat_map.shape[0],feat_map.shape[2]]).cuda()
        #TODO
        # Try to make this faster and more amenable to multi-GPU training.
        for i in range(feat_map.shape[2]):
            group = feat_map[:,:,i:,:] 
            group_activity_1[:,i] = self.group_activity(group)
        #zeros = torch.empty_like(group_activity_1)
        zeros = torch.zeros([group_activity_1.shape[0],group_activity_1.shape[1]]).cuda()
        L1 = torch.mean(F.pairwise_distance(group_activity_1,zeros,1))   # L1-norm

        group_activity_2 = torch.zeros([feat_map.shape[0],feat_map.shape[2]]).cuda()
        for j in reversed(range(feat_map.shape[2])):
            group = feat_map[:,:,:j+1,:]
            group_activity_2[:,j] = self.group_activity(group)
        L2 = torch.mean(F.pairwise_distance(group_activity_2,zeros,1)) 

        group_activity_3 = torch.zeros([feat_map.shape[0],feat_map.shape[3]]).cuda()
        for k in range(feat_map.shape[3]):
            group = feat_map[:,:,:,k:]
            group_activity_3[:,k] = self.group_activity(group)
        L3 = torch.mean(F.pairwise_distance(group_activity_3,zeros,1)) 

        group_activity_4 = torch.zeros([feat_map.shape[0],feat_map.shape[3]]).cuda()
        for l in reversed(range(feat_map.shape[3])):
            group = feat_map[:,:,:,:l+1]
            group_activity_4[:,l] = self.group_activity(group)
        L4 = torch.mean(F.pairwise_distance(group_activity_4,zeros,1))
        tot_loss = (L1.cuda() + L2.cuda() + L3.cuda() + L4.cuda())/4.0


        # More efficient implementation
        #TODO
        # There's some issue with this. The loss suddenly becomes zero. Most probably some
        # device/memory management thing.
        # Ignoring for now due to time constraints.
        #squared_feat_map = torch.mul(feat_map,feat_map)
        #squared_feat_map_channels = squared_feat_map.sum(dim=1)
        #
        #squared_feat_map_rows = squared_feat_map_channels.sum(dim=1)
        #m = squared_feat_map_rows.shape[1]
        #lr_group_norms = squared_feat_map_rows.cumsum(dim=1)**0.5
        #squared_feat_map_rows_reversed =\
        #torch.index_select(squared_feat_map_rows,1,torch.linspace(-1,-1*m,m).long().cuda() + m)
        #rl_group_norms = squared_feat_map_rows_reversed.cumsum(dim=1)**0.5

        #squared_feat_map_cols = squared_feat_map_channels.sum(dim=2)
        #n = squared_feat_map_cols.shape[1]
        #tb_group_norms = squared_feat_map_cols.cumsum(dim=1)**0.5
        #squared_feat_map_cols_reversed =\
        #torch.index_select(squared_feat_map_cols,1,torch.linspace(-1,-1*n,n).long().cuda() + n)
        #bt_group_norms = squared_feat_map_cols_reversed.cumsum(dim=1)**0.5

        #zeros_lr = torch.zeros([lr_group_norms.shape[0], lr_group_norms.shape[1]]).cuda()
        #zeros_tb = torch.zeros([tb_group_norms.shape[0], tb_group_norms.shape[1]]).cuda()
        #L1_alt = torch.mean(F.pairwise_distance(lr_group_norms,zeros_lr,1))
        #L2_alt = torch.mean(F.pairwise_distance(rl_group_norms,zeros_lr,1))
        #L3_alt = torch.mean(F.pairwise_distance(tb_group_norms,zeros_tb,1))
        #L4_alt = torch.mean(F.pairwise_distance(bt_group_norms,zeros_tb,1))

        #tot_loss_alt = (L1_alt.cuda() + L2_alt.cuda() + L3_alt.cuda() + L4_alt.cuda())/4.0

        #return tot_loss_alt, torch.log10(lr_group_norms+0.0000001), torch.log10(rl_group_norms+0.0000001), torch.log10(tb_group_norms+0.0000001), torch.log10(bt_group_norms+0.0000001)
        return tot_loss, torch.log10(group_activity_1), torch.log10(group_activity_2), torch.log10(group_activity_3), torch.log10(group_activity_4)


# Locality loss (group sparsity) over Class Activation Maps
class CAMLocalityLoss(torch.nn.Module):
    """
    Enforces small activation regions
    """
    def __init__(self):
        super(CAMLocalityLoss, self).__init__()
    def forward(self, cams):
        """
        Input: cams -> T x (CxHxW)  # Where H is the height of the feature map, W is the width,
        and C is the number of classes. T is the batch size.  CAM -> Class Activation Map
        Output: L = Locality Loss -> penalises activations with large spreads in the CAMs
        """
        squared_cams = torch.mul(cams,cams) #TxCxHxW

        #See LocalityLoss class for a better understanding of things

        # Get left-to-right and right-to-left groups and group activations
        squared_cams_rows = torch.sum(squared_cams, dim=2)  #TxCxW
        m = squared_cams_rows.shape[2]
        reversed_squared_cams_rows =\
        torch.index_select(squared_cams_rows,2,torch.linspace(-1,-1*m,m).long() + m)

        # L2 norm of grpups
        lr_groups = (torch.cumsum(squared_cams_rows, dim=2) + 1e-20)**0.5  #TxCxW
                                                                           # Adding 1e-20 to prevent NaN gradients
        # Loss for the lr_groups - L1 norm
        L1 = torch.mean(torch.sum(torch.sum(lr_groups, dim=2), dim=1))

        rl_groups = (torch.cumsum(reversed_squared_cams_rows, dim=2) + 1e-20)**0.5 
        L2 = torch.mean(torch.sum(torch.sum(rl_groups, dim=2), dim=1))

        squared_cams_cols = torch.sum(squared_cams, dim=3)  #TxCxH
        n = squared_cams_cols.shape[2]
        reversed_squared_cams_cols =\
        torch.index_select(squared_cams_cols,2,torch.linspace(-1,-1*n,n).long() + n)

        tb_groups = (torch.cumsum(squared_cams_cols, dim=2) + 1e-20)**0.5  #TxCxH
        L3 = torch.mean(torch.sum(torch.sum(tb_groups, dim=2), dim=1))

        bt_groups = (torch.cumsum(reversed_squared_cams_cols, dim=2) + 1e-20)**0.5 
        L4 = torch.mean(torch.sum(torch.sum(bt_groups, dim=2), dim=1))

        tot_loss = L1.cuda() + L2.cuda() + L3.cuda() + L4.cuda() 

        return tot_loss, torch.mean(torch.log10(lr_groups + 0.00001),dim=1), torch.mean(torch.log10(rl_groups + 0.00001),dim=1), torch.mean(torch.log10(tb_groups + 0.00001),dim=1), torch.mean(torch.log10(bt_groups + 0.00001),dim=1)


# This loss is an idea in progress and can be safely ignored for now
#class LocalityEntropyLoss(torch.nn.Module):
#    """
#    Enforces small activation regions
#    The idea is to follow a similar approach as MEL and BEL. The activation in 1 image should be
#    small (hence, lower entropy). However, averaged over a mini-batch, the mean activation should be
#    spread out. (And hence, higher entropy)
#    """
#    def __init__(self):
#        super(LocalityEntropyLoss, self).__init__()
#    def entropy(self, logits):
#        return -1.0*(F.softmax(logits,dim=0)*F.log_softmax(logits,dim=0)).sum()
#    def forward(self, feat_map):
#        """
#        Input: feat_map -> T x (HxWxD)  # Where H is the height of the feature map, W is the width,
#        and D is the depth. T is the batch size.
#        Output: L = Locality Entropy Loss MEL, and Locality Entropy Loss BEL
#        """
#        linearized_activations = feat_map.view(feat_map.size(0),-1)  # bs x (H*W*D)
#        sum1 = torch.zeros([linearized_activations.shape[0],1])
#        for t in range(linearized_activations.shape[0]):
#            sum1[t] = self.entropy(linearized_activations[t,:])
#        L1 = torch.mean(sum1)
#
#        average_linealized_activation = torch.mean(linearized_activations,dim=0)
#        L2 = -1.0*self.entropy(average_linealized_activation)
#        
#        return L1.cuda(), L2.cuda()


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
