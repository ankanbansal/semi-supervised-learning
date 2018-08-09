import numpy as np
import time
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helperFunctions import AverageMeter, accuracy, error

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as tv_utils


# Train the complete model (With all the losses)
def train_wsod_model(train_loader, model, criterion_list, optimizer, epoch, options, summary_writer):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_MEL = AverageMeter()
    losses_BEL = AverageMeter()
    total_losses = AverageMeter()

    criterion_cls = criterion_list[0]
    criterion_clust = criterion_list[1]

    # Set model to train mode
    model.train()

    end = time.time()
    for j, data in enumerate(train_loader):
        input_img_var = Variable(data[0].cuda(async=True))
        target_var = Variable(data[1].cuda(async=True))

        # model returns feature map, logits, and probabilities after applying softmax on logits
        logits, sm_output = model(input_img_var, options)
        class_with_max_prob = torch.argmax(sm_output,dim=1)

        # Calculate losses
        loss_0 = criterion_cls(logits, target_var)
        loss_2, loss_3 = criterion_clust(logits)

        loss = loss_0 + options['alpha']*loss_2 + options['beta']*loss_3

        # Add to tensorboard summary event
        summary_writer.add_scalar('loss/cls', loss_0.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/MEL', loss_2.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/BEL', loss_3.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/total', loss.item(), epoch*len(train_loader) + j)

        losses_cls.update(loss_0.item())
        losses_MEL.update(loss_2.item())
        losses_BEL.update(loss_3.item())
        total_losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
       
        if j%options['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f}) | '
                  'MEL Loss {MEL_loss.val:.4f} ({MEL_loss.avg:.4f}) | '
                  'BEL Loss {BEL_loss.val:.4f} ({BEL_loss.avg:.4f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(epoch, j,
                      len(train_loader), cls_loss=losses_cls, MEL_loss=losses_MEL, 
                      BEL_loss=losses_BEL, loss=total_losses, batch_time=batch_time))



def validate_model(val_loader, model, criterion, options):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1err = AverageMeter()

    model.eval()

    end = time.time()
    for j, data in enumerate(val_loader):
        input_img_var = Variable(data[0].cuda(async=True))
        target = data[1].cuda(async=True)
        target_var = Variable(target)

        logits, _ = model(input_img_var, options)

        loss = criterion(logits, target_var)

        prec1, prec5 = accuracy(logits.data, target, topK=(1,5))
        err = error(logits.data, target)
        losses_cls.update(loss.data[0], data[0].size(0))
        top1.update(prec1[0],data[0].size(0))
        top5.update(prec5[0],data[0].size(0))
        top1err.update(err[0],data[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # Why is this printing prec only in multiples of 5?
        if j%options['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) | '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        j, len(val_loader), batch_time=batch_time, loss=losses_cls, top1=top1,
                        top5=top5))
    
    #print('*Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, top1err.avg
