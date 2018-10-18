import numpy as np
import time
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helperFunctions import AverageMeter, accuracy

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as tv_utils

# Train a vanilla classification model
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

        # Optimization step
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


# Train the complete model (With all the losses)
# Use this if using the vanilla model for semi-supervised classification (WSODModel in models.py) or
# the single branch model with larger CAMs (WSODModel_LargerCAM_SameBranch in models.py)
def train_wsod_model(train_loader, model, criterion_list, optimizer, epoch, options, summary_writer):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_loc = AverageMeter()
    losses_MEL = AverageMeter()
    losses_BEL = AverageMeter()
    #losses_LEL_MEL = AverageMeter()
    #losses_LEL_BEL = AverageMeter()
    total_losses = AverageMeter()

    criterion_cls = criterion_list[0]
    criterion_loc = criterion_list[1]
    criterion_clust = criterion_list[2]
    #criterion_loc_ent = criterion_list[3]

    # Set model to train mode
    model.train()

    end = time.time()
    for j, data in enumerate(train_loader):
        input_img_var = Variable(data['image'].cuda(async=True))
        target_var = Variable(data['label'].cuda(async=True))

        # model returns feature map, logits, and probabilities after applying softmax on logits
        feat_map, logits, sm_output = model(input_img_var, options)
        class_with_max_prob = torch.argmax(sm_output,dim=1)

        # Calculate losses
        loss_0 = criterion_cls(logits, target_var)

        if options['CAM']:
            #CAM - remember to change loss in main.py
            weights = model.module.classifier.weight  # 1000x2208
            weights2 = weights.unsqueeze(2).unsqueeze(3)  # 1000x2208x1x1 #Make weights 4-D
            weights3 = weights2.repeat(1,1,feat_map.shape[2],feat_map.shape[3])  # 1000x2208x7x7 # Repeat 
            CAMs = torch.zeros([feat_map.shape[0],weights.shape[0],feat_map.shape[2],feat_map.shape[3]])
            CAMs_for_loss = torch.zeros([feat_map.shape[0],1,feat_map.shape[2],feat_map.shape[3]])
            for im_ind in range(CAMs.shape[0]):
                #CAMs[im_ind,:,:,:] = torch.mean(weights3*feat_map[im_ind], dim=1)
                CAMs[im_ind,:,:,:] = torch.sum(weights3*feat_map[im_ind], dim=1)  # Should ideally also divide by the sum of weights
                CAMs_for_loss[im_ind,:,:,:] = CAMs[im_ind,class_with_max_prob[im_ind],:,:]

            #Now apply locality loss on CAMs. On each class separately
            loss_1, g1, g2, g3, g4 = criterion_loc(CAMs.cuda(), sm_output)
        else:
            #Loc loss on feature map
            loss_1, g1, g2, g3, g4 = criterion_loc(feat_map)
            # activation decrease in g1 and g3
            # increase in g2 and g4

        loss_2, loss_3 = criterion_clust(logits)

        loss = loss_0 + options['gamma']*loss_1 + options['alpha']*loss_2 + options['beta']*loss_3

        # Add to tensorboard summary event
        summary_writer.add_scalar('loss/cls', loss_0.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/loc', loss_1.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/MEL', loss_2.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/BEL', loss_3.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/total', loss.item(), epoch*len(train_loader) + j)

        if options['hist_on'] and j%options['print_freq']==0:
            # Add histogram of the group activations        
            # Somehow very Time-intensive for CAM
            #try:
            #    #t1 = time.time()
            #    summary_writer.add_histogram('histogram/g1', g1.clone().cpu().data.numpy(),
            #            epoch*len(train_loader) + j, bins='auto')
            #    #t2 = time.time()
            #    summary_writer.add_histogram('histogram/g2', g2.clone().cpu().data.numpy(),
            #            epoch*len(train_loader) + j, bins='auto')
            #    #t3 = time.time()
            #    summary_writer.add_histogram('histogram/g3', g3.clone().cpu().data.numpy(),
            #            epoch*len(train_loader) + j, bins='auto')
            #    #t4 = time.time()
            #    summary_writer.add_histogram('histogram/g4', g4.clone().cpu().data.numpy(),
            #            epoch*len(train_loader) + j, bins='auto')
            #    #t5 = time.time()
            #    #if (t4-t3) > 5*(t2-t1):
            #    #    ipdb.set_trace()
            #except:
            #    print 'Cannot write histograms'
            #    ipdb.set_trace()

            # Add scalar for area
            #thresh = -5
            #thresholded_g1 = (g1>thresh).cpu().numpy()
            #thresholded_g2 = (g2>thresh).cpu().numpy()
            #thresholded_g3 = (g3>thresh).cpu().numpy()
            #thresholded_g4 = (g4>thresh).cpu().numpy()

            #heights = torch.zeros([g1.shape[0],1])
            #widths = torch.zeros([g1.shape[0],1])
            #areas = torch.zeros([g1.shape[0],1])
            #for s in range(g1.shape[0]):
            #    x0 = feat_map.shape[2] - next((i for i,x in enumerate(np.flip(thresholded_g1[s,:],0)) if x), 0)
            #    x1 = next((i for i,x in enumerate(thresholded_g2[s,:]) if x), 0)
            #    widths[s] = abs(x0 - x1)
            #    y0 = feat_map.shape[3] - next((i for i,x in enumerate(np.flip(thresholded_g3[s,:],0)) if x), 0)
            #    y1 = next((i for i,x in enumerate(thresholded_g4[s,:]) if x), 0)
            #    heights[s] = abs(y0 - y1)
            #    areas[s] = widths[s]*heights[s]
            #average_area = torch.mean(areas)
            #summary_writer.add_scalar('average_area/feature_map', average_area.item(), epoch*len(train_loader) + j)

            if j%(options['print_freq']*10) == 0:
                #TODO
                # Print CAMs instead of feat maps in case of CAMS
                if options['CAM']:
                    avg_feat_map = torch.mean(CAMs_for_loss,dim=1,keepdim=True)
                else:
                    avg_feat_map = torch.mean(feat_map,dim=1,keepdim=True)
                x = tv_utils.make_grid(avg_feat_map, normalize=True, scale_each=True)
                summary_writer.add_image('Image',x,epoch*len(train_loader) + j)

        losses_cls.update(loss_0.item())
        losses_loc.update(loss_1.item())
        losses_MEL.update(loss_2.item())
        losses_BEL.update(loss_3.item())
        #losses_LEL_MEL.update(loss_4.item())
        #losses_LEL_BEL.update(loss_5.item())
        total_losses.update(loss.item())

        # Do something like this to accumulate gradients to effectively increase the batch size.
        # This will be very useful for BEL
        # https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient-in-pytorch-i-e-iter-size-in-caffe-prototxt/2522/4
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        #if batch_time.val > 2.0:
        #    ipdb.set_trace()
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
                      MEL_loss=losses_MEL, BEL_loss=losses_BEL, loss=total_losses,
                      batch_time=batch_time))


# Train the complete model (With all the losses)
# Use this if you are using the model WSODModel_LargerCAM from models.py
# In this setting there is a separate branch for the locality loss and the main branch is only used
# for x-entropy, MEL, and NBEL
def train_wsod_model_larger_CAM(train_loader, model, criterion_list, optimizer, epoch, options, summary_writer):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_cls_large = AverageMeter()
    losses_loc = AverageMeter()
    losses_MEL = AverageMeter()
    losses_BEL = AverageMeter()
    total_losses = AverageMeter()

    criterion_cls = criterion_list[0]
    criterion_loc = criterion_list[1]
    criterion_clust = criterion_list[2]

    # Set model to train mode
    model.train()

    end = time.time()
    for j, data in enumerate(train_loader):
        input_img_var = Variable(data['image'].cuda(async=True))
        target_var = Variable(data['label'].cuda(async=True))

        # model returns feature map, logits, and probabilities after applying softmax on logits
        feat_map, logits_large, logits, sm_output_large = model(input_img_var, options)
        class_with_max_prob_large = torch.argmax(sm_output_large,dim=1)

        # Calculate losses
        loss_0 = criterion_cls(logits, target_var)
        loss_0_large = criterion_cls(logits_large, target_var)

        #CAM - remember to change loss in main.py
        weights = model.module.classifier1.weight  # 1000x1056
        weights2 = weights.unsqueeze(2).unsqueeze(3)  # 1000x1056x1x1 #Make weights 4-D
        weights3 = weights2.repeat(1,1,feat_map.shape[2],feat_map.shape[3])  # 1000x1056x14x14 # Repeat 
        CAMs = torch.zeros([feat_map.shape[0],weights.shape[0],feat_map.shape[2],feat_map.shape[3]])
        CAMs_for_loss = torch.zeros([feat_map.shape[0],1,feat_map.shape[2],feat_map.shape[3]])
        for im_ind in range(CAMs.shape[0]):
            CAMs[im_ind,:,:,:] = torch.sum(weights3*feat_map[im_ind], dim=1)  # Should ideally also divide by the sum of weights
            CAMs_for_loss[im_ind,:,:,:] = CAMs[im_ind,class_with_max_prob_large[im_ind],:,:]

        #Now apply locality loss on CAMs. On each class separately
        loss_1, g1, g2, g3, g4 = criterion_loc(CAMs.cuda(), sm_output_large)

        loss_2, loss_3 = criterion_clust(logits)

        loss = loss_0 + loss_0_large + options['gamma']*loss_1 + options['alpha']*loss_2 + options['beta']*loss_3

        # Add to tensorboard summary event
        summary_writer.add_scalar('loss/cls', loss_0.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/cls_large', loss_0_large.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/loc', loss_1.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/MEL', loss_2.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/BEL', loss_3.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/total', loss.item(), epoch*len(train_loader) + j)

        if options['hist_on'] and j%(options['print_freq']*10)==0:
            avg_feat_map = torch.mean(CAMs_for_loss,dim=1,keepdim=True)
            x = tv_utils.make_grid(avg_feat_map, normalize=True, scale_each=True)
            summary_writer.add_image('Image',x,epoch*len(train_loader) + j)

        losses_cls.update(loss_0.item())
        losses_cls_large.update(loss_0_large.item())
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
                  'Cls Loss Large {cls_loss_large.val:.4f} ({cls_loss_large.avg:.4f}) | '
                  'Loc Loss {loc_loss.val:.4f} ({loc_loss.avg:.4f}) | '
                  'MEL Loss {MEL_loss.val:.4f} ({MEL_loss.avg:.4f}) | '
                  'BEL Loss {BEL_loss.val:.4f} ({BEL_loss.avg:.4f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(epoch, j,
                      len(train_loader), cls_loss=losses_cls, cls_loss_large=losses_cls_large,
                      loc_loss=losses_loc, MEL_loss=losses_MEL, BEL_loss=losses_BEL, 
                      loss=total_losses, batch_time=batch_time))


# Train the model (With only clustering losses and gradient accumulation for larger
# effective bs)
# This larger effective batch size is achieved by updating the model only after passing multiple
# batches through the network
def train_wsod_model_multiple(train_loader, model, criterion_list, optimizer, epoch, options, summary_writer):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_MEL = AverageMeter()
    losses_BEL = AverageMeter()

    criterion_cls = criterion_list[0]
    criterion_clust = criterion_list[2]

    model.train()

    end = time.time()

    total_loss = 0.0
    optimizer.zero_grad()
    for j, data in enumerate(train_loader):
        input_img_var = Variable(data['image'].cuda(async=True))
        target_var = Variable(data['label'].cuda(async=True))

        # model returns feature map, logits, and probabilities after applying softmax on logits
        feat_map, logits, _ = model(input_img_var, options)

        # Calculate losses
        loss_0 = criterion_cls(logits, target_var)
        loss_2, loss_3 = criterion_clust(logits)
        loss = (loss_0 + options['alpha']*loss_2 + options['beta']*loss_3)/options['accum_batches']  # Ideally should divide by accum_batches
        #optimizer.zero_grad()
        loss.backward()
        
        total_loss += loss

        if j%options['accum_batches'] < 0.5:
            optimizer.step()
            optimizer.zero_grad()
            total_loss = 0.0

        # Add to tensorboard summary event
        summary_writer.add_scalar('loss/cls', loss_0.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/MEL', loss_2.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/BEL', loss_3.item(), epoch*len(train_loader) + j)
        #summary_writer.add_scalar('loss/Total', loss.item(), epoch*len(train_loader) + j)
        summary_writer.add_scalar('loss/Total', total_loss, epoch*len(train_loader) + j)

        losses_cls.update(loss_0.item())
        losses_MEL.update(loss_2.item())
        losses_BEL.update(loss_3.item())

        # Do something like this to accumulate gradients to effectively increase the batch size.
        # This will be very useful for BEL
        # https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient-in-pytorch-i-e-iter-size-in-caffe-prototxt/2522/4
        # Optimization step

        batch_time.update(time.time() - end)
        end = time.time()
       
        if j%options['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f}) | '
                  'MEL Loss {MEL_loss.val:.4f} ({MEL_loss.avg:.4f}) | '
                  'BEL Loss {BEL_loss.val:.4f} ({BEL_loss.avg:.4f}) | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(epoch, j,
                      len(train_loader), cls_loss=losses_cls,
                      MEL_loss=losses_MEL, BEL_loss=losses_BEL,
                      batch_time=batch_time))


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
        #_, _, logits, _ = model(input_img_var, options)

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
