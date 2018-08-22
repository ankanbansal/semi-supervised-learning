import os
import numpy as np
import json
import ipdb
import argparse

#
import models
import dataLoader
from losses import get_loss
from helperFunctions import save_checkpoint, adjust_learning_rate, plot_CAMs

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from tensorboardX import SummaryWriter


# Main file for everything
def argparser():
    parser = argparse.ArgumentParser(description='Weakly Supervised Object Detection')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    #TODO
    # Choose densenet201 for the final implementation
    parser.add_argument('--base_arch', type=str, default='densenet161',
            choices=['densenet161','densenet169','densenet201','resnet152'], 
            help='Which model to use as the base architecture')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--test_checkpoint', type=str, default=None, help='Enter filename.')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--val_batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sup_json_file', type=str, default='./Data/train_whole_sup_50k_tot_50k.json')
    parser.add_argument('--unsup_json_file', type=str, default='./Data/train_whole_sup_0_tot_150k.json')
    parser.add_argument('--val_on', type=bool, default=True)
    parser.add_argument('--val_json_file', type=str, default='./Data/train_whole_sup_50k_tot_50k.json')
    parser.add_argument('--train_img_dir', type=str,
            default='/efs2/data/imagenet/train/')
    parser.add_argument('--val_img_dir', type=str, default='/efs2/data/imagenet/train/')
    parser.add_argument('--sup_to_tot_ratio', type=float, default=0.25)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=1)
    options = vars(parser.parse_args())
    return options

if __name__ == "__main__":
    options = argparser()
    best_avg_prec = 0.0
    is_best = False
    #model = models.BasicClassificationModel(options)
    model = models.WSODModel_LargerCAM(options)

    model = nn.DataParallel(model).cuda()
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    print 'Creating data loaders...'
    train_loader, val_loader = dataLoader.weighted_loaders(options)
    print 'Created data loaders'

    writer = SummaryWriter(options['log_dir'])
    print 'Starting validate...'
    print 'Loading checkpoint {} ...'.format(options['test_checkpoint'])
    checkpoint = torch.load(options['test_checkpoint'])
    model.load_state_dict(checkpoint['state_dict'])
    plot_CAMs(val_loader, model, options)
    print 'Average Precision: ', prec1
