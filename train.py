import numpy as np
import time
import ipdb
from helper_functions import AverageMeter

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def train_model(train_loader, model, criterion, optimizer, epoch, options):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for j, data in enumerate(train_loader):
        input_img_var = Variable(data['image'].cuda(async=True))
        target_var = Variable(data[''].cuda(async=True))

        data_time.update(time.time() - end)

        _____ = model(input_img_var, _____)
        loss = criterion(____, target_var)

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


def validate_model():
# Training iterations
