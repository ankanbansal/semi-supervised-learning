import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = float(self.sum)/float(self.count)

def save_checkpoint(state,filename='./checkpoints_temp/checkpoint.pth.tar',is_best=False):
    torch.save(state,filename)
    #if is_best:
    #    shutil.copyfile(filename,'./checkpoints/model_best.pth.tar')

def adjust_learning_rate(optimizer,epoch,model_options,d):
    """Sets the lr to the initial lr decayed by 10 every d epochs"""
    lr = model_options['learning_rate']*(0.1**(epoch//d))
    print 'Learning rate: ', lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
        res.append(correct_k.mul_(100.0 / float(batch_size)))
    return res

def error(output, target):
    maxk = 1
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() #transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #element-wise equality
    incorrect = 1 - correct
    incorrect_k = incorrect.view(-1).float().sum(0)
    res = incorrect_k.mul_(100.0 / float(batch_size))
    return res

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
