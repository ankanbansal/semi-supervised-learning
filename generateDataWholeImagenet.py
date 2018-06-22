import os, sys
import json
import argparse
import random
import ipdb
import re

def argparser():
    parser = argparse.ArgumentParser(description="Generating json files from xml")
    parser.add_argument('--label_dict_file', type=str, default='./Data/label_dict.json')
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--img_dir_root', type=str,
            default='/efs/data/imagenet/')
    parser.add_argument('--img_set_dir', type=str,
            default='/efs/data/imagenet/train_meta/subsets/meta/')
    parser.add_argument('--img_set', type=str, default='train')

    options = vars(parser.parse_args())
    return options

if __name__ == "__main__":
    options = argparser()

    label_dict = json.load(open(options['label_dict_file']))
    if options['img_set'] == 'train':
        options['file_list'] = os.path.join(options['img_set_dir'], 'imagenet_train.lst')
        options['save_file'] = os.path.join('./Data/only_annotated/', 'train_whole_' + str(options['ratio']) + '.json')
    elif options['img_set'] == 'val':
        options['file_list'] = os.path.join(options['img_set_dir'], 'imagenet_val.lst')
        options['save_file'] = os.path.join('./Data/', 'val_whole.json')
    else:
        sys.exit('Not a valid image set')

    file_list = []
    labels = []
    with open(options['file_list']) as f:
        for line in f:
            file_list.append(re.split(r'\t+',line.strip())[-1])
            labels.append(int(re.split(r'\t+',line.strip())[1]))

    if options['img_set'] == 'train':
        all_images = []
        for k,fname in enumerate(file_list):
            if k%10000 == 0:
                print k
            img = {}
            img['image_name'] = fname
            # Take only 1/ratio samples for class supervision
            if k%options['ratio'] == 0:
                if labels[k] == label_dict[fname.split('/')[0]]:
                    img['label'] = labels[k]
                else:
                    ipdb.set_trace()
                all_images.append(img)
            else:
                img['label'] = -1000
            #all_images.append(img)
        json.dump(all_images,open(options['save_file'],'w'))
    else:
        all_images = []
        for k,fname in enumerate(file_list):
            if k %1000 == 0:
                print k
            img = {}
            img['image_name'] = fname
            img['label'] = labels[k]
            all_images.append(img)
        json.dump(all_images,open(options['save_file'],'w'))
