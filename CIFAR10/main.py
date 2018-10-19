import os
import numpy as np
import json
# import ipdb
import argparse

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
    parser.add_argument('--log_dir', type=str, default='./sup_4k_tot_50k/temp_logs/')
    parser.add_argument('--base_arch', type=str, default='densenet_cifar',
            choices=['densenet_cifar','densenet121'], 
            help='Which model to use as the base architecture')
    parser.add_argument('--mode', type=str, default='train', choices=['train','test','validate'])
    parser.add_argument('--type', type=str, default='cls_clust',
            choices=['cls','cls_clust','cls_MEL','cls_BEL','only_clust','cls_reg','cls_MEL_reg','cls_clust_reg','all'])
    parser.add_argument('--resume', type=str, default=None, help='Enter filename.')
    parser.add_argument('--test_checkpoint', type=str, default=None, help='Enter filename.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_transformations', type=int, default=4)  # If you are using
                                                                       # regularized_weighted_loaders,
                                                                       # then batch_size should be a
                                                                       # multiple of
                                                                       # num_transformations
    parser.add_argument('--reg_distance_type', type=str, default='Euclidean', choices=['Euclidean',
        'cosine', 'KL'])
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='/scratch1/Rainforest/data/')
    parser.add_argument('--val_on', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='./sup_4k_tot_50k/temp_checkpoints/')
    parser.add_argument('--num_sup', type=int, default=4000)
    parser.add_argument('--num_unsup', type=int, default=46000)
    parser.add_argument('--sup_to_tot_ratio', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=0.8) # Multiplier for MEL
    parser.add_argument('--beta', type=float, default=0.5) # Multiplier for NBEL
    parser.add_argument('--delta', type=float, default=0.05) # Multiplier for Reg
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--lr_step', type=int, default=150)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=10)

    options = vars(parser.parse_args())
    return options

if __name__ == "__main__":
    options = argparser()

    # The following return loss classes
    criterion_cls = get_loss(loss_name='CE') # Cross-entropy loss
    criterion_clust = get_loss(loss_name='ClusterLoss') # MEL + BEL
    criterion_reg = get_loss(loss_name='STLoss') # Regularization with Stochastic Transformations
                                                 # loss

    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    if options['mode'] == 'train':
        if options['type'] == 'cls_clust':
            # Use only classification and clustering (MEL + BEL)
            options['gamma'] = 0
            options['delta'] = 0
        elif options['type'] == 'cls':
            # Use only classification
            options['alpha'] = 0
            options['beta'] = 0
            options['gamma'] = 0
            options['delta'] = 0
        elif options['type'] == 'cls_MEL':
            # Use only classification and MEL
            options['beta'] = 0
            options['gamma'] = 0
            options['delta'] = 0
        elif options['type'] == 'cls_BEL':
            # Use only classification and BEL
            options['alpha'] = 0
            options['gamma'] = 0
            options['delta'] = 0
        elif options['type'] == 'only_clust':
            options['gamma'] = 0
            options['delta'] = 0
            #TODO
            # Make classification weight zero too
        elif options['type'] == 'cls_reg':
            options['alpha'] = 0
            options['beta'] = 0
            options['gamma'] = 0
        elif options['type'] == 'cls_MEL_reg':
            options['beta'] = 0
            options['gamma'] = 0
        elif options['type'] == 'cls_clust_reg':
            options['gamma'] = 0

        val_accuracies = []
        val_errors = []
        for run in range(options['runs']):
            log_dir = os.path.join(options['log_dir'], 'run_{}'.format(run))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir)
            best_avg_prec = 0.0
            best_avg_err = 100.0
            is_best = False
            model = models.WSODModel(options)
            model = nn.DataParallel(model).cuda()
            # Resume from checkpoint
            if options['resume']:
                if os.path.isfile(options['resume']):
                    print 'Loading checkpoint {}...'.format(options['resume'])
                    checkpoint = torch.load(options['resume'])
                    options['start_epoch'] = checkpoint['epoch']
                    best_avg_prec = checkpoint['best_avg_prec']
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    print 'File {} does not exist'.format(options['resume'])

            print 'Creating data loaders...'
            if 'reg' in options['type']:
                train_loader, val_loader = dataLoader.regularized_weighted_loaders(options)
            else:
                train_loader, val_loader = dataLoader.weighted_loaders(options)
            print 'Created data loaders'

            optimizer = torch.optim.SGD(model.parameters(), options['learning_rate'], nesterov=True,
                    momentum=0.9, dampening=0, weight_decay=0.0001)

            print 'Start training for run: ', run
            for epoch in range(options['start_epoch'], options['epochs']):
                # Validate
                if options['val_on']:
                    avg_prec, avg_err = validate_model(val_loader, model, criterion_cls, options)
                    is_best = avg_prec > best_avg_prec
                    if is_best:
                        print 'Best model till now: ', epoch
                        best_avg_prec = max(avg_prec, best_avg_prec)
                        best_avg_err = min(avg_err, best_avg_err)
                        print 'Saving checkpoint after ', epoch, ' epochs...'
                        save_dir = os.path.join(options['save_dir'], 'run_{}'.format(run))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_checkpoint({'epoch': epoch+1,
                                         'base_arch': options['base_arch'],
                                         'state_dict': model.state_dict(),
                                         'best_avg_prec': best_avg_prec},
                                        filename = os.path.join(save_dir,
                                            'checkpoint_{}_epoch_{}.pth.tar'.format(options['type'],epoch)),
                                        is_best=is_best)

                    writer.add_scalar('validation/prec1', avg_prec, epoch)
                    writer.add_scalar('validation/err', avg_err, epoch)

                # Adjust learning rate. Divide learning rate by 10 every d epochs.
                d = options['lr_step']
                adjust_learning_rate(optimizer, epoch, options, d)

                print 'Training for epoch:', epoch
                train_wsod_model(train_loader,model,[criterion_cls,criterion_clust,criterion_reg],optimizer,epoch,options,writer)

            val_accuracies.append(best_avg_prec)
            val_errors.append(best_avg_err)
            writer.close()

        json.dump(val_errors,open(os.path.join(options['log_dir'],'val_errors.json'),'w'))

        print 'Errors: ', val_errors
        print 'Mean: ', np.mean(val_errors), 'STD: ', np.std(val_errors)
    else:
        print 'Starting validation...'
        print 'Loading checkpoint {} ...'.format(options['test_checkpoint'])
        checkpoint = torch.load(options['test_checkpoint'])
        model.load_state_dict(checkpoint['state_dict'])
        prec1, err = validate_model(val_loader, model, criterion_cls, options)
        print 'Average Precision: ', prec1
