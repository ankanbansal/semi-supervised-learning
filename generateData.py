import os, sys
import xml.etree.ElementTree as ET
import json
import argparse
import random
import ipdb

def argparser():
    parser = argparse.ArgumentParser(description="Generating json files from xml")
    parser.add_argument('--label_dict_file', type=str, default='./Data/label_dict.json')
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--img_dir_root', type=str,
            default='/efs/data/weakly-detection-data/imagenet-detection/ILSVRC/Data/CLS-LOC/')
    parser.add_argument('--annot_dir_root', type=str,
            default='/efs/data/weakly-detection-data/imagenet-detection/ILSVRC/Annotations/CLS-LOC/')
    parser.add_argument('--img_set_dir', type=str,
            default='/efs/data/weakly-detection-data/imagenet-detection/ILSVRC/ImageSets/CLS-LOC/')
    parser.add_argument('--img_set', type=str, default='train')

    options = vars(parser.parse_args())
    return options

if __name__ == "__main__":
    options = argparser()
    """
    The following is for generating object level data. We don't really need that.
    """
#    if options['img_set'] == 'train':
#        options['file_list'] = os.path.join(options['img_set_dir'], 'train_loc.txt')
#        options['save_file'] = os.path.join('./Data/', 'train_objects.json')
#    elif options['img_set'] == 'val':
#        options['file_list'] = os.path.join(options['img_set_dir'], 'val.txt')
#        options['save_file'] = os.path.join('./Data/', 'val_objects.json')
#    else:
#        sys.exit('Not a valid image set')
#
#    print 'Objects will be saved to ', options['save_file']
#
#    file_list = []
#    with open(options['file_list']) as f:
#        for line in f:
#            file_list.append(line.strip().split(" ")[0])
#
#    k = 0
#    all_objects = []
#    for fname in file_list:
#        if k%100 == 0:
#            print k
#            #ipdb.set_trace()
#        annot_fname = os.path.join(options['annot_dir_root'],options['img_set'],fname + '.xml')
#
#        tree = ET.parse(annot_fname)
#        root = tree.getroot()
#        img_name = str(root.find('filename').text) + '.JPEG'
#        img_dims = root.find('size')
#        img_width = int(img_dims.find('width').text)
#        img_height = int(img_dims.find('height').text)
#        for obj in root.findall('object'):
#            new_object = {}
#            new_object['img_name'] = img_name
#            new_object['img_width'] = img_width
#            new_object['img_height'] = img_height
#            new_object['name'] = str(obj.find('name').text)
#
#            bnd_box = obj.find('bndbox')
#            new_object['xmin'] = int(bnd_box.find('xmin').text)
#            new_object['ymin'] = int(bnd_box.find('ymin').text)
#            new_object['xmax'] = int(bnd_box.find('xmax').text)
#            new_object['ymax'] = int(bnd_box.find('ymax').text)
#
#            all_objects.append(new_object)
#        k += 1
#
#    json.dump(all_objects, open(options['save_file'],'w'))

    """
    The following is for image level data. This is more useful
    """
    label_dict = json.load(open(options['label_dict_file']))
    if options['img_set'] == 'train':
        options['file_list'] = os.path.join(options['img_set_dir'], 'train_loc.txt')
        options['save_file'] = os.path.join('./Data/', 'train_' + str(options['ratio']) + '.json')
    elif options['img_set'] == 'val':
        options['file_list'] = os.path.join(options['img_set_dir'], 'val.txt')
        options['save_file'] = os.path.join('./Data/', 'val.json')
    else:
        sys.exit('Not a valid image set')

    file_list = []
    with open(options['file_list']) as f:
        for line in f:
            file_list.append(line.strip().split(" ")[0])

    if options['img_set'] == 'train':
        k = 0
        all_images = []
        for fname in file_list:
            if k%10000 == 0:
                print k
            img = {}
            img['image_name'] = fname + '.JPEG'
            if k%options['ratio'] == 0:
                img['label'] = label_dict[fname.split('/')[0]]
            else:
                img['label'] = -1000
            all_images.append(img)
            k += 1
        json.dump(all_images,open(options['save_file'],'w'))
    else:
        all_images = []
        k = 0
        for fname in file_list:
            if k %1000 == 0:
                print k
            img = {}
            img['image_name'] = fname + '.JPEG'
            annot_fname = os.path.join(options['annot_dir_root'],options['img_set'],fname+'.xml')
            tree = ET.parse(annot_fname)
            root = tree.getroot()
            obj = root.find('object')
            class_name = str(obj.find('name').text)
            img['label'] = label_dict[class_name]
            all_images.append(img)
            k += 1
        json.dump(all_images,open(options['save_file'],'w'))
