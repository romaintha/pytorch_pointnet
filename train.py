import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from fastprogress import master_bar, progress_bar

from datasets import ShapeNetDataset, PointMNIST
from model.pointnet import ClassificationPointNet, SegmentationPointNet
from utils import plot_losses, plot_accuracies


MODELS = {
    'classification': ClassificationPointNet,
    'segmentation': SegmentationPointNet
}

DATASETS = {
    'shapenet': ShapeNetDataset,
    'mnist': PointMNIST
}


def train(dataset,
          dataset_folder,
          task,
          number_of_points,
          batch_size,
          epochs,
          learning_rate,
          output_folder,
          number_of_workers,
          model_checkpoint):
    train_dataset = DATASETS[dataset](dataset_folder,
                                      task=task,
                                      number_of_points=number_of_points)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers)
    test_dataset = DATASETS[dataset](dataset_folder,
                                     task=task,
                                     train=False,
                                     number_of_points=number_of_points)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=number_of_workers)

    if task == 'classification':
        model = ClassificationPointNet(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=train_dataset.POINT_DIMENSION)
    elif task == 'segmentation':
        model = SegmentationPointNet(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
                                     point_dimension=train_dataset.POINT_DIMENSION)
    else:
        raise Exception('Unknown task !')

    if torch.cuda.is_available():
        model.cuda()
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mb = master_bar(range(epochs))

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    with open(os.path.join(output_folder, 'training_log.csv'), 'w+') as fid:
        fid.write('train_loss,test_loss,train_accuracy,test_accuracy\n')

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in mb:
        epoch_train_loss = []
        epoch_train_acc = []
        batch_number = 0
        for data in progress_bar(train_dataloader, parent=mb):
            batch_number += 1
            points, targets = data
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()
            if points.shape[0] <= 1:
                continue
            optimizer.zero_grad()
            model = model.train()
            preds, feature_transform = model(points)
            if task == 'segmentation':
                preds = preds.view(-1, train_dataset.NUM_SEGMENTATION_CLASSES)
                targets = targets.view(-1)

            identity = torch.eye(feature_transform.shape[-1])
            if torch.cuda.is_available():
                identity = identity.cuda()
            regularization_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1))
            )
            loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
            epoch_train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
            preds = preds.data.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            if task == 'classification':
                accuracy = corrects.item() / float(batch_size)
            elif task == 'segmentation':
                accuracy = corrects.item() / float(batch_size*number_of_points)
            epoch_train_acc.append(accuracy)
            mb.child.comment = 'train loss: %f, train accuracy: %f' % (np.mean(epoch_train_loss),
                                                                       np.mean(epoch_train_acc))
        epoch_test_loss = []
        epoch_test_acc = []
        for batch_number, data in enumerate(test_dataloader):
            points, targets = data
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()
            model = model.eval()
            preds, feature_transform = model(points)
            if task == 'segmentation':
                preds = preds.view(-1, train_dataset.NUM_SEGMENTATION_CLASSES)
                targets = targets.view(-1)
            loss = F.nll_loss(preds, targets)
            epoch_test_loss.append(loss.cpu().item())
            preds = preds.data.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            if task == 'classification':
                accuracy = corrects.item() / float(batch_size)
            elif task == 'segmentation':
                accuracy = corrects.item() / float(batch_size*number_of_points)
            epoch_test_acc.append(accuracy)

        mb.write('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'
                 % (epoch,
                    np.mean(epoch_train_loss),
                    np.mean(epoch_test_loss),
                    np.mean(epoch_train_acc),
                    np.mean(epoch_test_acc)))
        if test_acc and np.mean(epoch_test_acc) > np.max(test_acc):
            torch.save(model.state_dict(), os.path.join(output_folder, 'shapenet_%s_model.pth' % task))

        with open(os.path.join(output_folder, 'training_log.csv'), 'a') as fid:
            fid.write('%s,%s,%s,%s,%s\n' % (epoch,
                                            np.mean(epoch_train_loss),
                                            np.mean(epoch_test_loss),
                                            np.mean(epoch_train_acc),
                                            np.mean(epoch_test_acc)))
        train_loss.append(np.mean(epoch_train_loss))
        test_loss.append(np.mean(epoch_test_loss))
        train_acc.append(np.mean(epoch_train_acc))
        test_acc.append(np.mean(epoch_test_acc))

    plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_folder, 'loss_plot.png'))
    plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_folder, 'accuracy_plot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['shapenet', 'mnist'], help='dataset to train on')
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2500, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=1, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')

    args = parser.parse_args()

    train(args.dataset,
          args.dataset_folder,
          args.task,
          args.number_of_points,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.output_folder,
          args.number_of_workers,
          args.model_checkpoint)
