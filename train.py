import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

from fastprogress import master_bar, progress_bar

from datasets import ShapeNetDataset
from model.pointnet import ClassificationPointNet, SegmentationPointNet


def train_shapenet(dataset_folder,
                   task,
                   number_of_points,
                   batch_size,
                   epochs,
                   learning_rate,
                   output_folder,
                   number_of_workers,
                   model_checkpoint):
    train_dataset = ShapeNetDataset(dataset_folder,
                                    task=task,
                                    number_of_points=number_of_points)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers)
    test_dataset = ShapeNetDataset(dataset_folder,
                                   task=task,
                                   is_training=False,
                                   number_of_points=number_of_points)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=number_of_workers)

    if task == 'classification':
        model = ClassificationPointNet(num_classes=len(train_dataset.folders_to_classes_mapping))
    elif task == 'segmentation':
        model = SegmentationPointNet(num_classes=len(train_dataset.folders_to_classes_mapping))
    else:
        raise Exception('Unknown task !')
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
    val_acc = []
    for epoch in mb:
        epoch_train_loss = []
        epoch_train_acc = []
        batch_number = 0
        for data in progress_bar(train_dataloader, parent=mb):
            batch_number += 1
            points, targets = data
            if task == 'classification':
                targets = targets[:, 0]
            points, targets = points.cuda(), targets.cuda()
            if points.shape[0] <= 1:
                continue
            optimizer.zero_grad()
            model = model.train()
            preds = model(points)
            if task == 'segmentation':
                preds = preds.view(-1, len(train_dataset.folders_to_classes_mapping))
                targets = targets.view(-1)
            loss = F.nll_loss(preds, targets)
            epoch_train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
            pred_choice = preds.data.max(1)[1]
            corrects = pred_choice.eq(targets.data).cpu().sum()
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
            if task == 'classification':
                targets = targets[:, 0]
            points, targets = points.cuda(), targets.cuda()
            model = model.eval()
            preds = model(points)
            if task == 'segmentation':
                preds = preds.view(-1, len(train_dataset.folders_to_classes_mapping))
                targets = targets.view(-1)
            loss = F.nll_loss(preds, targets)
            epoch_test_loss.append(loss.cpu().item())
            pred_choice = preds.data.max(1)[1]
            corrects = pred_choice.eq(targets.data).cpu().sum()
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
        if val_acc and np.mean(epoch_test_acc) > np.max(val_acc):
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
        val_acc.append(np.mean(epoch_test_acc))

    fig = plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, test_loss, 'b', label='Test loss')
    plt.title('Training and test loss')
    plt.legend()
    fig.savefig(os.path.join(output_folder, 'loss_plot.png'))

    fig = plt.figure()
    plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.legend()
    fig.savefig(os.path.join(output_folder, 'accuracy_plot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['shapenet'], help='dataset to train on')
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

    if args.dataset == 'shapenet':
        train_shapenet(args.dataset_folder,
                       args.task,
                       args.number_of_points,
                       args.batch_size,
                       args.epochs,
                       args.learning_rate,
                       args.output_folder,
                       args.number_of_workers,
                       args.model_checkpoint)
