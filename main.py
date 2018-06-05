import os
import numpy as np
import json
import ipdb
import argparse

import models
import dataLoader
#from train import train_basic_model#, validate_model
from train import train_wsod_model#, validate_model
from losses import get_loss
from helperFunctions import save_checkpoint, adjust_learning_rate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from tensorboardX import SummaryWriter


# Main file for everything
def argparser():
    parser = argparse.ArgumentParser(description='Weakly Supervised Object Detection')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--base_arch', type=str, default='densenet161',
            choices=['densenet161','densenet169','densenet201','resnet152'], 
            help='Which model to use as the base architecture')
    parser.add_argument('--mode', type=str, default='train', choices=['train','test','validate'])
    parser.add_argument('--resume', type=str, default=None, help='Want to start from a checkpoint? Enter filename.')
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_json_file', type=str, default='./Data/train_1.json')
    parser.add_argument('--val_on', type=bool, default=False)
    parser.add_argument('--val_json_file', type=str, default='')
    parser.add_argument('--train_img_dir', type=str,
            default='/efs/data/weakly-detection-data/imagenet-detection/ILSVRC/Data/CLS-LOC/train/')
    parser.add_argument('--val_img_dir', type=str, default='')
    parser.add_argument('--num_blocks', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=32)
    #TODO
    # Cross-validate the hyper-parameters to obtain the best values
    parser.add_argument('--lmbda', type=float, default=0.1)
    parser.add_argument('--gamma_1', type=float, default=0.01)
    parser.add_argument('--gamma_2', type=float, default=100.0)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=100)

    options = vars(parser.parse_args())
    return options

if __name__ == "__main__":
    options = argparser()
    best_metric = 0.0
    is_best = False
    writer = SummaryWriter(options['log_dir'])
    #model = models.BasicClassificationModel(options)
    model = models.WSODModel(options)
    criterion_cls = get_loss(options,loss_name='CE')
    criterion_loc = get_loss(options,loss_name='LocalityLoss')
    criterion_clust = get_loss(options,loss_name='ClusterLoss')

    model = nn.DataParallel(model).cuda()
    torch.multiprocessing.set_sharing_strategy('file_system')

    if options['resume']:
        if os.path.isfile(options['resume']):
            print 'Loading checkpoint {} ...'.format(options['resume'])
            checkpoint = torch.load(options['resume'])
            options['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print 'File {} does not exist'.format(options['resume'])

    cudnn.benchmark = True

    print 'Creating data loaders...'
    train_loader, val_loader = dataLoader.loaders(options)
    print 'Created data loaders'

    optimizer = torch.optim.SGD(model.parameters(), options['learning_rate'])

    print 'Start training...'
    for epoch in range(options['start_epoch'], options['epochs']):
        adjust_learning_rate(optimizer, epoch, options)
        print 'Training for epoch:', epoch

        #train_basic_model(train_loader,model,criterion,optimizer,epoch,options)
        train_wsod_model(train_loader,model,[criterion_cls,criterion_loc,criterion_clust],optimizer,epoch,options,writer)

        # Validate
        if options['val_on']:
            metric = validate_model()
            is_best = metric > best_metric
            best_metric = max(metric,best_metric)

        save_checkpoint({'epoch': epoch+1,
                         'base_arch': options['base_arch'],
                         'state_dict': model.state_dict(),
                         'best_metric': best_metric},
                        filename = 'checkpoints/checkpoint_epoch_{}.pth.tar'.format(epoch),
                        is_best=is_best)
    writer.close()
