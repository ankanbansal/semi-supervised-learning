import numpy as np
import time
import ipdb
from helperFunctions import AverageMeter

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def train_basic_model(train_loader, model, criterion, optimizer, epoch, options):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for j, data in enumerate(train_loader):
        input_img_var = Variable(data['image'].cuda(async=True))
        target_var = Variable(data['label'].cuda(async=True))

        data_time.update(time.time() - end)

        logits = model(input_img_var)
        loss = criterion(logits, target_var)

        losses.update(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if j%options['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(epoch, j,
                      len(train_loader), loss=losses, batch_time=batch_time, data_time=data_time))


def train_wsod_model(train_loader, model, criterion_list, optimizer, epoch, options, summary_writer):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_loc = AverageMeter()
    losses_MEL = AverageMeter()
    losses_BEL = AverageMeter()
    total_losses = AverageMeter()

    criterion_cls = criterion_list[0]
    criterion_loc = criterion_list[1]
    criterion_clust = criterion_list[2]

    model.train()

    end = time.time()
    for j, data in enumerate(train_loader):
        input_img_var = Variable(data['image'].cuda(async=True))
        target_var = Variable(data['label'].cuda(async=True))

        feat_map, logits, sm_output = model(input_img_var, options)
        loss_0 = criterion_cls(logits, target_var)
        loss_1 = criterion_loc(feat_map)
        loss_2, loss_3 = criterion_clust(logits)  # Make sure that you have to pass logits and not sm_output
        loss = loss_0 + options['gamma']*loss_1 + options['alpha']*loss_2 + options['beta']*loss_3

        summary_writer.add_scalar('loss/cls', loss_0.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/loc', loss_1.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/MEL', loss_2.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/BEL', loss_3.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/total', loss.item(), epoch*len(train_loader) + j)

        losses_cls.update(loss_0.item())
        losses_loc.update(loss_1.item())
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
                  'Loc Loss {loc_loss.val:.4f} ({loc_loss.avg:.4f}) | '
                  'MEL Loss {MEL_loss.val:.4f} ({MEL_loss.avg:.4f}) | '
                  'BEL Loss {BEL_loss.val:.4f} ({BEL_loss.avg:.4f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(epoch, j,
                      len(train_loader), cls_loss=losses_cls, loc_loss=losses_loc,
                      MEL_loss=losses_MEL, BEL_loss=losses_BEL, loss=total_losses, batch_time=batch_time))


def validate_model(val_loader, model, criterion, options):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for j, data in enumerate(val_loader):
        target = data['label'].cuda(async=True)
        input_img_var = Variable(data['image'].cuda(async=True))
        target_var = Variable(target)

        _, logits, _ = model(input_img_var, options)
        loss = criterion(logits, target_var)

        prec1, prec5 = accuracy(logits.data, target, topK=(1,5))
        losses_cls.update(loss.data[0], data['image'].size(0))
        top1.update(prec1[0],data['image'].size(0))
        top5.update(prec5[0],data['image'].size(0))

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
    return top1.avg


def accuracy(output, target, topK=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topK)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() #transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #element-wise equality

    res = []
    for k in topK:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))  # This leads to prediction in multiples of 100.0/batch_size?
                                                        # But why?
    return res
