import torch
import numpy as np
import shutil

class BBox(object):
    def __init__(self,x1,y1,x2,y2):
        if x1>x2: x1,x2 = x2,x1
        if y1>y2: y1,y2 = y2,y1
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def area(self):
        return (self.x2 - self.x1)*(self.y2 - self.y1)

    def intersectionBox(self,other):
        overlap = BBox(0,0,0,0)
        if self.x2 < other.x1 or self.x1 > other.x2:
            return overlap
        overlap.x1 = self.x1 if (self.x1>other.x1) else other.x1
        overlap.x2 = self.x2 if (self.x2<other.x2) else other.x2
        overlap.y1 = self.y1 if (self.y1>other.y1) else other.y1
        overlap.y2 = self.y2 if (self.y2<other.y2) else other.y2
        return overlap

    def unionArea(self,other):
        intersection_box = self.intersectionBox(other)
        return self.area() + other.area() - intersection_box.area()

    def IOU(self,other):
        intersection_box = self.intersectionBox(other)
        return float(intersection_box.area())/float(self.unionArea(other))


class JsonProgress(object):
    def __init__(self):
        self.count = 0

    def __call__(self, obj):
        self.count += 1
        sys.stdout.write("\r%8d" % self.count)
        return obj


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
    if is_best:
        shutil.copyfile(filename,'./checkpoints/model_best.pth.tar')

def adjust_learning_rate(optimizer,epoch,model_options):
    """Sets the lr to the initial lr decayed by 10 every 2 epochs"""
    lr = model_options['learning_rate']*(0.1**(epoch//2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
