import json
import os
import csv
import torch.utils.data as data
import torch
from torchvision.datasets import MNIST

import numpy as np

from utils import transform_2d_img_to_point_cloud


class ShapeNetDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 16
    NUM_SEGMENTATION_CLASSES = 50

    POINT_DIMENSION = 3

    PER_CLASS_NUM_SEGMENTATION_CLASSES = {
        'Airplane': 4,
        'Bag': 2,
        'Cap': 2,
        'Car': 4,
        'Chair': 4,
        'Earphone': 3,
        'Guitar': 3,
        'Knife': 2,
        'Lamp': 4,
        'Laptop': 2,
        'Motorbike': 6,
        'Mug': 2,
        'Pistol': 3,
        'Rocket': 3,
        'Skateboard': 3,
        'Table': 3,
    }

    def __init__(self,
                 dataset_folder,
                 number_of_points=2500,
                 task='classification',
                 train=True):
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        assert task in ['classification', 'segmentation']
        self.task = task
        self.train = train

        category_file = os.path.join(self.dataset_folder, 'synsetoffset2category.txt')
        self.folders_to_classes_mapping = {}
        self.segmentation_classes_offset = {}

        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            offset_seg_class = 0
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
                self.segmentation_classes_offset[row[1]] = offset_seg_class
                offset_seg_class += self.PER_CLASS_NUM_SEGMENTATION_CLASSES[row[0]]

        if self.train:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        self.files = [(f.split('/')[1], f.split('/')[2]) for f in filenames]

    def __getitem__(self, index):
        folder, file = self.files[index]
        point_file = os.path.join(self.dataset_folder,
                                  folder,
                                  'points',
                                  '%s.pts' % file)
        segmentation_label_file = os.path.join(self.dataset_folder,
                                               folder,
                                               'points_label',
                                               '%s.seg' % file)
        point_cloud_class = self.folders_to_classes_mapping[folder]
        if self.task == 'classification':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class)
        elif self.task == 'segmentation':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class,
                                     segmentation_label_file=segmentation_label_file,
                                     segmentation_classes_offset=self.segmentation_classes_offset[folder])

    def __len__(self):
        return len(self.files)

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     point_cloud_class=None,
                     segmentation_label_file=None,
                     segmentation_classes_offset=None):
        point_cloud = np.loadtxt(point_file).astype(np.float32)
        if number_of_points:
            sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
            point_cloud = point_cloud[sampling_indices, :]
        point_cloud = torch.from_numpy(point_cloud)
        if segmentation_label_file:
            segmentation_classes = np.loadtxt(segmentation_label_file).astype(np.int64)
            if number_of_points:
                segmentation_classes = segmentation_classes[sampling_indices]
            segmentation_classes = segmentation_classes + segmentation_classes_offset -1
            segmentation_classes = torch.from_numpy(segmentation_classes)
            return point_cloud, segmentation_classes
        elif point_cloud_class is not None:
            point_cloud_class = torch.tensor(point_cloud_class)
            return point_cloud, point_cloud_class
        else:
            return point_cloud


class PointMNIST(MNIST):

    NUM_CLASSIFICATION_CLASSES = 10

    POINT_DIMENSION = 2

    def __init__(self, *args, **kwargs):
        kwargs.pop('task')
        self.number_of_points = kwargs.pop('number_of_points')
        kwargs['download'] = True
        super(PointMNIST, self).__init__(*args, **kwargs)
        self.transform = transform_2d_img_to_point_cloud

    def __getitem__(self, index):
        img, target = super(PointMNIST, self).__getitem__(index)
        sampling_indices = np.random.choice(img.shape[0], self.number_of_points)
        img = img[sampling_indices, :].astype(np.float32)
        img = torch.tensor(img)
        return img, target
