import os
import json
import csv

if __name__ == "__main__":
    label_dict = {}
    class_file = '/efs/data/imagenet/train_meta/subsets/meta/imagenet_classes.csv'
    with open(class_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_dict[row['class_name']] = int(row['class'])
    json.dump(label_dict,open('./Data/label_dict.json','w'))
