# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

from __future__ import print_function

import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}
    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    # TODO: Adjust data_dir according to where **you** stored the data
    def __init__(self, split, size, data_dir='VOCdevkit/VOC2007/'):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.size = size
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()
        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.Resize((self.size,self.size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    @classmethod
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)
    
    def get_labels_and_weights(self, index):
        fpath = os.path.join(self.ann_dir, index + '.xml')
        tree = ET.parse(fpath)
        root = tree.getroot()
        labels = np.zeros(20)
        weights = np.ones(20)
        name = None
        for child in root:
            if child.tag != 'object':
                continue
            for attr in child:
                if attr.tag == 'name':
                    name = attr.text
                    labels[self.INV_CLASS[name]] = 1
                if attr.tag == 'difficult' and attr.text == '1':
                    weights[self.INV_CLASS[name]] = 1
        return [labels, weights]

    def preload_anno(self):
        """
        :return: a list of lables. each element is in the form of [class, weight],
         where both class and weight are a numpy array in shape of [20],
        """
        label_list = []
        for index in self.index_list:
            label_list.append(self.get_labels_and_weights(index))
        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element
        image: FloatTensor in shape of (C, H, W) in scale [-1, 1].
        label: LongTensor in shape of (Nc, ) binary label
        weight: FloatTensor in shape of (Nc, ) difficult or not.
        """
        findex = self.index_list[index]
        label, wgt = self.get_labels_and_weights(findex)
        fpath = os.path.join(self.img_dir, findex + '.jpg')
        # TODO: insert your code here. hint: read image, find the labels and weight.
        img = Image.open(fpath)
        #img = np.array(Image.open(fpath))
        #img = img - np.array([[[123.68, 116.78, 103.94]]])
        #img = Image.fromarray(img.astype(np.uint8))
        
        image = self.transform(img)
        label = torch.FloatTensor(label)
        wgt = torch.FloatTensor(wgt)
        return image, label, wgt
