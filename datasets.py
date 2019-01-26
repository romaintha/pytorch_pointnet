import json
import os
import csv
import torch.utils.data as data
import torch

import numpy as np


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 dataset_folder,
                 number_of_points=2500,
                 task='classification',
                 is_training=True):
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        assert task in ['classification', 'segmentation']
        self.task = task
        self.is_training = is_training

        category_file = os.path.join(self.dataset_folder, 'synsetoffset2category.txt')
        self.folders_to_classes_mapping = {}
        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
        if is_training:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        self.files = [(f.split('/')[1], f.split('/')[2]) for f in filenames]

    def __getitem__(self, index):
        folder, file = self.files[index]
        point_cloud = np.loadtxt(os.path.join(self.dataset_folder,
                                              folder,
                                              'points',
                                              '%s.pts' % file)).astype(np.float32)
        sampling_indices = np.random.choice(point_cloud.shape[0], self.number_of_points)
        point_cloud = torch.from_numpy(point_cloud[sampling_indices, :])
        if self.task == 'classification':
            point_cloud_class = torch.tensor([self.folders_to_classes_mapping[folder]])
            return point_cloud, point_cloud_class
        elif self.task == 'segmentation':
            segmentation_classes = np.loadtxt(os.path.join(self.dataset_folder,
                                                           folder,
                                                           'points_label',
                                                           '%s.seg' % file)).astype(np.int64)
            segmentation_classes = torch.from_numpy(segmentation_classes[sampling_indices])
            return point_cloud, segmentation_classes

    def __len__(self):
        return len(self.files)
