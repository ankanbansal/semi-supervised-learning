import os
import numpy as np
import json 
import ipdb
import argparse

#
import models
import dataLoader
from train import train_wsod_model, validate_model
from losses import get_loss
from helperFunctions import save_checkpoint, adjust_learning_rate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from tensorboardX import SummaryWriter


# Main file for everything
def argparser():
    parser = argparse.ArgumentParser(description='Semi-Supervised Learning')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--base_arch', type=str, default='densenet121',
            choices=['densenet121','resnet18'], 
            help='Which model to use as the base architecture')
    parser.add_argument('--mode', type=str, default='train', choices=['train','test','validate'])
    parser.add_argument('--type', type=str, default='cls_clust',
            choices=['cls','cls_loc','cls_clust','all', 'cls_MEL', 'cls_BEL', 'cls_MEL_loc', 'cls_MEL_LELMEL',
                'cls_MEL_LEL', 'cls_clust_LELMEL', 'cls_clust_LEL','only_clust'])
    parser.add_argument('--resume', type=str, default=None, help='Want to start from a checkpoint? Enter filename.')
    parser.add_argument('--test_checkpoint', type=str, default=None, help='Enter filename.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='/efs2/data/')
    parser.add_argument('--val_on', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='./sup_4k_tot_50k/checkpoints/')
    parser.add_argument('--dataset_size', type=int, default=50000)
    parser.add_argument('--num_sup', type=int, default=4000)
    parser.add_argument('--sup_to_tot_ratio', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.8)  # Multiplier for MEL
    parser.add_argument('--beta', type=float, default=0.8)  # Multiplier for BEL
    parser.add_argument('--accum_batches', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_step', type=int, default=30)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=130)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=10)

    options = vars(parser.parse_args())
    return options

if __name__ == "__main__":
    options = argparser()
    best_avg_prec = 0.0
    is_best = False
    model = models.WSODModel(options)

    # The following return loss classes
    criterion_cls = get_loss(loss_name='CE')  # Cross-entropy loss
    criterion_clust = get_loss(loss_name='ClusterLoss')  # MEL + BEL

    model = nn.DataParallel(model).cuda()
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Resume from checkpoint
    if options['resume']:
        if os.path.isfile(options['resume']):
            print 'Loading checkpoint {}...'.format(options['resume'])
            checkpoint = torch.load(options['resume'])
            options['start_epoch'] = checkpoint['epoch']
            #options['start_epoch'] = 39
            best_avg_prec = checkpoint['best_avg_prec']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print 'File {} does not exist'.format(options['resume'])

    cudnn.benchmark = True

    print 'Creating data loaders...'
    train_loader, val_loader = dataLoader.weighted_loaders(options)
    print 'Created data loaders'

    optimizer = torch.optim.SGD(model.parameters(), options['learning_rate'], nesterov=True,
            momentum=0.9, dampening=0, weight_decay=0.0001)

    if options['mode'] == 'train':
        writer = SummaryWriter(options['log_dir'])
        if options['type'] == 'cls_clust':
            # Use only classification and clustering (MEL + BEL)
            options['gamma'] = 0
            options['nu'] = 0
            options['mu'] = 0
        elif options['type'] == 'cls':
            # Use only classification
            options['gamma'] = 0
            options['alpha'] = 0
            options['beta'] = 0
            options['nu'] = 0
            options['mu'] = 0
        elif options['type'] == 'cls_MEL':
            # Use only classification and MEL
            options['gamma'] = 0
            options['beta'] = 0
            options['nu'] = 0
            options['mu'] = 0
        elif options['type'] == 'cls_BEL':
            # Use only classification and MEL
            options['alpha'] = 0
            options['gamma'] = 0
            options['nu'] = 0
            options['mu'] = 0
        elif options['type'] == 'only_clust':
            options['gamma'] = 0
            #TODO
            # Make classification weight zero too

        print 'Start training...'
        for epoch in range(options['start_epoch'], options['epochs']):
            # Validate
            if options['val_on']:
                avg_prec = validate_model(val_loader, model, criterion_cls, options)
                is_best = avg_prec > best_avg_prec
                if is_best:
                    print 'Best model till now: ', epoch
                    best_avg_prec = max(avg_prec,best_avg_prec)
                    print 'Saving checkpoint after ', epoch, ' epochs...'
                    save_checkpoint({'epoch': epoch+1,
                                     'base_arch': options['base_arch'],
                                     'state_dict': model.state_dict(),
                                     'best_avg_prec': best_avg_prec},
                                    filename = options['save_dir'] + 'checkpoint_{}_epoch_{}.pth.tar'.format(options['type'],epoch),
                                    is_best=is_best)
                writer.add_scalar('validation/prec1', avg_prec, epoch)

            # Adjust learning rate. Divide learning rate by 10 every d epochs.
            d = options['lr_step']
            adjust_learning_rate(optimizer, epoch, options, d)
            # Can also look into torch.optim.lr_scheduler and
            # torch.optim.lr_scheduler.ReduceLROnPlateau

            print 'Training for epoch:', epoch
            #train_basic_model(train_loader,model,criterion,optimizer,epoch,options)
            train_wsod_model(train_loader,model,[criterion_cls,criterion_clust],optimizer,epoch,options,writer)

        writer.close()
    else:
        print 'Starting validate...'
        print 'Loading checkpoint {} ...'.format(options['test_checkpoint'])
        checkpoint = torch.load(options['test_checkpoint'])
        model.load_state_dict(checkpoint['state_dict'])
        prec1 = validate_model(val_loader, model, criterion_cls, options)
        print 'Average Precision: ', prec1
