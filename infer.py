import argparse
import random

import numpy as np

import torch

import open3d

from model.pointnet import ClassificationPointNet, SegmentationPointNet
from datasets import ShapeNetDataset, PointMNISTDataset

MODELS = {
    'classification': ClassificationPointNet,
    'segmentation': SegmentationPointNet
}

DATASETS = {
    'shapenet': ShapeNetDataset,
    'mnist': PointMNISTDataset
}


def infer(dataset,
          model_checkpoint,
          point_cloud_file,
          task):
    if task == 'classification':
        num_classes = DATASETS[dataset].NUM_CLASSIFICATION_CLASSES
    elif task == 'segmentation':
        num_classes = DATASETS[dataset].NUM_SEGMENTATION_CLASSES
    model = MODELS[task](num_classes=num_classes,
                         point_dimension=DATASETS[dataset].POINT_DIMENSION)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(model_checkpoint))

    points = DATASETS[dataset].prepare_data(point_cloud_file)
    points = torch.tensor(points)
    if torch.cuda.is_available():
        points = points.cuda()
    points = points.unsqueeze(dim=0)
    model = model.eval()
    preds, feature_transform = model(points)
    if task == 'segmentation':
        preds = preds.view(-1, num_classes)
    preds = preds.data.max(1)[1]

    points = points.cpu().numpy().squeeze()
    preds = preds.cpu().numpy()

    if task == 'classification':
        print('Detected class: %s' % preds)
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((49,1))])
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(points)
        open3d.draw_geometries([pcd])
    elif task == 'segmentation':
        colors = [(random.randrange(256)/255, random.randrange(256)/255, random.randrange(256)/255)
                  for _ in range(num_classes)]
        rgb = [colors[p] for p in preds]
        rgb = np.array(rgb)

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(points)
        pcd.colors = open3d.Vector3dVector(rgb)
        open3d.draw_geometries([pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['shapenet', 'mnist'], type=str, help='dataset to train on')
    parser.add_argument('model_checkpoint', type=str, help='dataset to train on')
    parser.add_argument('point_cloud_file', type=str, help='path to the point cloud file')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')

    args = parser.parse_args()

    infer(args.dataset,
          args.model_checkpoint,
          args.point_cloud_file,
          args.task)
