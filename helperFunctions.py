import torch
import numpy as np
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generic Bounding Box class
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


# Not working somehow. Don't have the enthu to figure out why
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

def plot_CAMs(val_loader, model, options):
    model.eval()
    #upsampling = torch.nn.UpsamplingBilinear2d(size=(224,224))
    upsampling = torch.nn.Upsample((224,224), mode='bilinear')

    for j, data in enumerate(val_loader):
        if j % 10 == 0:
            print j
            target = data['label'].cuda(async=True)
            input_img_var = Variable(data['image'].cuda(async=True))
            target_var = Variable(target)

            # model returns feature map, logits, and probabilities after applying softmax on logits
            feat_map, logits, sm_output = model(input_img_var, options)

            class_with_max_prob = torch.argmax(sm_output,dim=1)
            weights = model.module.classifier.weight  # 1000x2208
            weights2 = weights.unsqueeze(2).unsqueeze(3)  # 1000x2208x1x1 #Make weights 4-D
            weights3 = weights2.repeat(1,1,feat_map.shape[2],feat_map.shape[3])  # 100x2208x7x7 # Repeat 
            CAMs = torch.zeros([feat_map.shape[0],weights.shape[0],feat_map.shape[2],feat_map.shape[3]])
            CAMs_for_loss = torch.zeros([feat_map.shape[0],1,feat_map.shape[2],feat_map.shape[3]])
            for im_ind in range(CAMs.shape[0]):
                #CAMs[im_ind,:,:,:] = torch.mean(weights3*feat_map[im_ind], dim=1)
                CAMs[im_ind,:,:,:] = torch.sum(weights3*feat_map[im_ind], dim=1)  # Should ideally also divide by the sum of weights
                CAMs_for_loss[im_ind,:,:,:] = CAMs[im_ind,class_with_max_prob[im_ind],:,:]

            upsampled_CAMs = upsampling(CAMs_for_loss).cuda()

            #TODO
            # Should we normalize CAMs - Yes
            # See: https://github.com/philipperemy/tensorflow-class-activation-mapping/blob/master/class_activation_map.py
            max_val = upsampled_CAMs.max()
            min_val = upsampled_CAMs.min()
            upsampled_CAMs = (upsampled_CAMs - min_val)/(max_val - min_val)

            images_with_cams = torch.cat([input_img_var, upsampled_CAMs], 1)

            #ipdb.set_trace()

            # Overlay CAMs on Images
            #x = tv_utils.make_grid(images_with_cams, normalize=True, scale_each=True) 
            # TensorboardX does not support RGBA!!
            #summary_writer.add_image('Activations',images_with_cams,j)

            for k in range(images_with_cams.shape[0]):
                plt.figure(1)
                plt.subplot(121)
                plt.imshow(input_img_var[k].cpu().transpose(0,2).transpose(0,1).detach().numpy())
                plt.title('Target Class: {}'.format(target[k]))
                plt.subplot(122)
                plt.imshow(1-input_img_var[k].cpu().transpose(0,2).transpose(0,1).detach().numpy())
                plt.imshow(upsampled_CAMs[k].cpu().transpose(0,2).transpose(0,1).detach().numpy().squeeze(axis=2),
                        cmap=plt.cm.jet, alpha=0.9, interpolation='nearest', vmin=0, vmax=1)
                #plt.imshow(images_with_cams[k].cpu().transpose(0,2).transpose(0,1).detach().numpy())
                plt.title('Predicted Class: {}'.format(class_with_max_prob[k]))

                cmap_file_name = 'cams/{}_{}.png'.format(j,k)
                plt.savefig(cmap_file_name)
                plt.close()

            #plt.ioff()

            #k = 0
            #for cam, img in zip(upsampled_CAMs, input_img_var):
            #    k += 1
            #    plt.imshow(img.cpu().transpose(0,2).transpose(0,1).numpy())
            #    plt.imshow(cam.cpu().transpose(0,2).transpose(0,1).numpy(), cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
            #    cmap_file_name = 'cams/{}.png'.format(k)
            #    plt.savefig(cmap_file_name)
            #    plt.close()

            #ipdb.set_trace()

