import os
import json

if __name__ == "__main__":
    root_dir = '/efs/data/weakly-detection-data/imagenet-detection/ILSVRC/Data/CLS-LOC/train'
    sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))]
    print len(sub_dirs)
    label_dict = {}
    for j,d in enumerate(sub_dirs):
        label_dict[d] = j
    json.dump(label_dict,open('./Data/label_dict.json','w'))
