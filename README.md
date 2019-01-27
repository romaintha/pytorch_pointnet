# Pytorch PointNet 
A Pytorch implementation of the PointNet network.
 
Reference: ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", Qi et al](https://arxiv.org/abs/1612.00593)

## Getting started
### Setting up
Install the dependencies using Conda:
```
conda create --name pytorch_pointnet --file spec-file.txt
```

### Available datasets
* ```shapenet```: download the dataset 
[here](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

More soon

### Training
Use the following script for training
```
python train.py dataset dataset_folder task output_folder 
   --number_of_points 2500
   --batch_size 32
   --epochs 50
   --learning_rate 0.001
   --number_of_workers 4
   --model_checkpoint
```

where:
* ```dataset```: is one of the available datasets (e.g. ```shapenet```)
* ```dataset_folder```: is the path to the root dataset folder
* ```task```: is either ```classification``` or ```segmentation```
* ```output_folder```: is the output_folder path where the training logs and model 
checkpoints will be stored
* ```number_of_points```: is the amount of points per cloud
* ```batch_size```: is the batch size 
* ```epochs```: is the number of training epochs
* ```learning_rate```: is the optimizer learning rate
* ```number_of_workers```: is the number of workers used by the data loader
* ```model_checkpoint```: is the path to a checkpoint that is loaded 
before the training begins.

### Infer
Use the following script for inference:
```
python infer.py dataset model_checkpoint point_cloud_file task 
```

where:
* ```dataset```: is one of the available datasets (e.g. ```shapenet```)
* ```model_checkpoint```: is the path to a checkpoint that is loaded 
before the inference begins.
* ```point_cloud_file```: is the path to the point cloud file to run the inference on.
* ```task```: is either ```classification``` or ```segmentation```

This will also output a 3d visualization of the point cloud.