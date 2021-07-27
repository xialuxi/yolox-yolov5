#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch
import cv2
import numpy as np
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from .yolo_classes import YOLO_CLASSES
from glob import glob

IMAGE_EXT = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']

class YOLODataset(Dataset):
    """
    yolo dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        train_images="images/train",
        val_images="images/val",
        train=True,
        img_size=(416, 416),
        preproc=None,
    ):
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "CVAT")
        self._classes = YOLO_CLASSES
        self.class_to_ind =  dict(zip(YOLO_CLASSES, range(len(YOLO_CLASSES))))
        self.data_dir = data_dir
        self.train_images_dir = train_images
        self.val_images_dir = val_images
        self.img_size = img_size
        self.preproc = preproc

        if train:
            self.images_list = self.get_image_files(data_dir, self.train_images_dir)
        else:
            self.images_list = self.get_image_files(data_dir, self.val_images_dir)

        #for debug
        #self.images_list = self.images_list[0:100]

        self.labels_list = self.img2label_paths(self.images_list)




    def __len__(self):
        return len(self.images_list)

    def load_anno(self, index):
        #one_image = self.train_images_list[index]
        lb_file = self.labels_list[index]
        with open(lb_file, 'r') as f:
            l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        if len(l):
            assert l.shape[1] == 5, 'labels require 5 columns each'
            assert (l >= 0).all(), 'negative labels'
            assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
            assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
        cls = l[:, 0]
        cls = np.expand_dims(cls, axis=1)
        box = l[:, 1:]
        return np.concatenate((box, cls), axis = 1)

    def pull_item(self, index):
        img_file = self.images_list[index]
        img = cv2.imread(img_file)
        assert img is not None
        # load anno
        h, w, c = img.shape
        res = self.load_anno(index)
        res[:, 0:4] = self.convert(w, h, res[:, 0:4])
        img_info = (h, w)
        return img, res, img_info, index

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

 #   @staticmethod
 #   def collate_fn(batch):
 #       img, label, img_info, img_id = zip(*batch)  # transposed
 #       return torch.stack(img, 0), torch.cat(label, 0), torch.from_numpy(np.array(img_info)), torch.from_numpy(np.array(img_id))

    def img2label_paths(self, img_paths):
        # Define label paths as a function of image paths
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]

    def get_image_files(self, path, dir):
        img_path = os.path.join(path, dir)
        all_file_list = []
        for ext in IMAGE_EXT:
            img_list = glob(os.path.join(img_path, '*' + ext))
            all_file_list.extend(img_list)
        return all_file_list

    def convert(self, w, h , box):
        box = box * [w, h, w, h]
        x1 = box[:, 0] - 0.5 * box[:, 2]
        y1 = box[:, 1] - 0.5 * box[:, 3]
        x2 = box[:, 0] + 0.5 * box[:, 2]
        y2 = box[:, 1] + 0.5 * box[:, 3]
        x1 = np.expand_dims(x1, axis=1)
        y1 = np.expand_dims(y1, axis=1)
        x2 = np.expand_dims(x2, axis=1)
        y2 = np.expand_dims(y2, axis=1)
        return np.concatenate((x1, y1, x2, y2), axis = 1)
