# AutoLR: Automatic Learning Rate Setting #
[Paper]()
## Status ##
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![maintenance](https://img.shields.io/badge/maintained%3F-yes-brightgreen.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![python](https://img.shields.io/badge/python-v3.7-blue)](https://www.python.org/downloads/release/python-370/)
![size](https://img.shields.io/github/repo-size/MathieuTuli/AutoLR)

## Table of Contents ##
- [Introduction](#introduction)
- [License](#license)
- [Citing AutoLR](#citing-autolr)
- [Requirements](#requirements)
  - [Software](#software)
  - [Hardware](#hardware)
- [Installation](#installation)
  - [Python Package](#python-package)
    * [Repository Cloning](#repository-cloning)
  - [Unpackaged Python](#unpackaged-python)
- [Usage](#usage)
  - [Running the Code](#running-the-code)
  - [Config File](#config-file)
    * [Available Datasets for Training](#available-datasets-for-training)
    * [Available Models for Training](#available-models-for-training)
    * [Optimizer Method](#optimizer-method)
    * [Learning Rate Scheduler](#learning-rate-scheduler)
    * [Initial Learning Rate](#initial-learning-rate)
    * [Early Stopping Threshold](#early-stopping-threshold)
    * [Optimizer Arguments](#optimizer-arguments)
    * [Scheduler Arguments](#scheduler-arguments)
    * [Number of Training Trials](#number-of-training-trials)
    * [Max Epochs](#max-epochs)
    * [Mini-Batch Size](#mini-batch-size)
    * [Early Stopping Patience](#early-stopping-patience)
    * [Power](#power)
  - [Training Outputs](#training-outputs)
    * [XLSX Output](#xlsx-output)
    * [Checkpoints](#checkpoints)
- [Common Issues (running list)](#common-issues--running-list-)
- [TODO](#todo)
- [Pytest](#pytest)

### Introduction ###
[Paper]()


This repository contains a [PyTorch](https://pytorch.org/) implementation of AutoLR.

### License ###
AutoLR is released under the MIT License (refer to the [LICENSE](LICENSE) file for more information)
|Permissions|Conditions|Limitations|
|---|---|---|
|![license](https://img.shields.io/badge/-%20-brightgreen) Commerical use|![license](https://img.shields.io/badge/-%20-blue) License and Copyright Notice|![license](https://img.shields.io/badge/-%20-red) Liability|
|![license](https://img.shields.io/badge/-%20-brightgreen) Distribution| | ![license](https://img.shields.io/badge/-%20-red) Warranty|
|![license](https://img.shields.io/badge/-%20-brightgreen) Modification | | |
|![license](https://img.shields.io/badge/-%20-brightgreen) Private Use| | |

### Citing AutoLR ###
```text
@misc{
}
```
### Requirements ###
#### Software ####
We use `Python 3.7`

Per [requirements.txt](requirements.txt), the following Python packages are required:
```text
-e .
future==0.18.2
joblib==0.16.0
memory-profiler==0.57.0
numpy==1.19.1
pandas==1.1.1
Pillow==7.2.0
psutil==5.7.2
python-dateutil==2.8.1
pytz==2020.1
PyYAML==5.3.1
scikit-learn==0.23.2
scipy==1.5.2
six==1.15.0
threadpoolctl==2.1.0
torch==1.6.0
torchvision==0.7.0
tqdm==4.48.2
```

**NOTE** that in order to satisfy `torch==1.6.0` the following (soft) Nvidia requirements need to be met:
- CUDA Version: `CUDA 10.2`
- CUDA Driver Version: `r440`
- CUDNN Version: `7.6.4-7.6.5`
For more information, refer to the [cudnn-support-matrix](https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html).

Refer to the [PyTorch installation guide](https://pytorch.org/) for information how to install PyTorch. We do not guarantee proper function of this code using different versions of PyTorch or CUDA.

#### Hardware ####
- GPU
  - At least 4 GB of GPU memory is required
- Memory
  - At least 8 GB of internal memory is required
Naturally, the memory requirements is scaled relative to current dataset being used and mini-batch sizes, we state these number using the CIFAR10 and CIFAR100 dataset.

### Installation ###
There are two versions of the AdaS code contained in this repository.
1. a python-package version of the AdaS code, which can be `pip`-installed.
2. a static python module (unpackaged), runable as a script.

#### Python Package ####

---

##### Repository Cloning #####
After cloning the repository (`git clone https://github.com/MathieuTuli/AutoLR.git`), simply run
```console
python setup.py build
python setup.py install
```
or
```console
pip install .
```
If you will be making changes and wish to not have to reinstall the package each time, run
```console
pip install -e .
```

Note that `pip install -e .` is a command included in the [requirements.txt](requirements.txt) file.

#### Unpackaged Python ####

---
Ensure first that you have the requirements installed per [requirements.txt](requirements.txt)

You can run the code by running
```console
python train.py --...
```
Where `--...` represents the options for training (see below)

Similarly, the `train.py` file could be run from an IDE. We leave specifying arguments up to the user if run this way. Note a quick and dirty way is to alter the default argument values found in `train.py`.

### Usage ###
Moving forward, I will refer to console usage of this library. IDE usage is no different. Training options are split two ways:
1. First, all environment/infrastructure options (GPU usage, output paths, etc.) is specified using arguments.
2. Second, all training specific options (network, dataset, hyper-parameters, etc.) is specified using a configuration file.
#### Running the Code ####
To start training, you must first define your training configuration file. An example can be found at [src/autolr/config.yaml](src/autolr/config.yaml). Finally, run
```console
python -m adas train --config path-to-config.yaml
```
Where you specify the path of your `config.yaml` file. Note the following additional options for autolr:
```console
usage: __main__.py train [-h] [--config CONFIG] [--data DATA]
                         [--output OUTPUT] [--checkpoint CHECKPOINT]
                         [--resume RESUME] [--root ROOT]
                         [--save-freq SAVE_FREQ] [--cpu] [--gpu GPU]
                         [--multiprocessing-distributed] [--dist-url DIST_URL]
                         [--dist-backend DIST_BACKEND]
                         [--world-size WORLD_SIZE] [--rank RANK]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Set configuration file path: Default = 'config.yaml'
  --data DATA           Set data directory path: Default = '.autolr-data'
  --output OUTPUT       Set output directory path: Default = '.autolr-output'
  --checkpoint CHECKPOINT
                        Set checkpoint directory path: Default = '.autolr-checkpoint'
  --resume RESUME       Set checkpoint resume path: Default = None
  --root ROOT           Set root path of project that parents all others:
                        Default = '.'
  --save-freq SAVE_FREQ
                        Checkpoint epoch save frequency: Default = 25
  --cpu                 Flag: CPU bound training: Default = False
  --gpu GPU             GPU id to use: Default = 0
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training: Default = False
  --dist-url DIST_URL   url used to set up distributed training:Default =
                        'tcp://127.0.0.1:23456'
  --dist-backend DIST_BACKEND
                        distributed backend: Default = 'nccl'
  --world-size WORLD_SIZE
                        Number of nodes for distributed training: Default = -1
  --rank RANK           Node rank for distributed training: Default = -1
```
#### Config File ####
In the following sections we list the configuration options available to the user. Note that we also classify the configuration options into the following categories:
- Application Specific
  - dataset
  - network
  - optimizer
  - scheduler
- Suggested Tune
   - init_lr
   - early_stop_threshold
   - optimizer_kwargs
   - scheduler_kwargs
- Suggested Default
   - n_trials
   - max_epochs
   - mini_batch_size
   - num_workers
   - loss
   - early_stop_patience
   - p

The **Suggested Default** parameters are ones we have preset and suggest not be altered too much. Naturally, the user may change them at their discretion.

The **Suggested Tune** parameters are highly recommended to be tuned, and are very application specific.

The **Application Specific** parameters then are simply ones that the user must change to do what they want (what dataset, model, learning algorithm, etc.)


#### Available Datasets for Training ####
---
**yaml identifier: dataset**
Currently the following datasets are supported:
- CIFAR10
- CIFAR100
- ImageNet

#### Available Models for Training ####

---
**yaml identifier: network**
All models used can be found in [src/autolr/models](src/adas/models). They are a combination of [PyTorch](https://github.com/pytorch/pytorch) and [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) models.
- AlexNet
- DenseNet201 | DenseNet169 | DenseNet161 | DenseNet121 | DenseNet121CIFAR
- GoogLeNet
- InceptionV3
- MNASNet_0_5 | MNASNet_0_75 | MNASNet_1 | MNASNet_1_3
- MobileNetV2
- ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 | ResNext50 | ResNext101 | ResNet18CIFAR | ResNet34CIFAR
- ResNeXtCIFAR
- WideResNet50 | WideResNet101
- ShuffleNetV2_0_5 | ShuffleNetV2_1 | ShuffleNetV2_1_5 | ShuffleNetV2_2
- SqueezeNet_1 | SqueezeNet_1_1
- VGG11 | VGG11_BN | VGG13 | VGG13_BN | VGG16 | VGG16_BN | VGG19 | VGG19_BN
- VGG16CIFAR
- EfficientNetB4 | EfficientNetB0CIFAR


#### Optimizer Method ####

---
**yaml identifier: optimizer**

Options:
- SGD
- NAG
- AdaM
- AdaGrad
- RMSProp
- AdaDelta
- AdaBound
- AMSBound
- AdaMax
- AdaMod
- AdaShift
- NAdam
- NosAdam
- NovoGrad
- PAdam
- RAdam
- SPS
- SLS
- LaProp
- LearningRateDropout

#### Learning Rate Scheduler ####

---
**yaml identifier: scheduler**

Options:
- AdaS (Note that `SGD` must be specified as the `optimizer`)
- StepLR
- CosineAnnealingWarmRestarts
- OneCycleLR

#### Initial Learning Rate ####

---
**yaml identifier: init_lr**

Initial learning rate for the optimizer method. Note that specifying 'auto' will run the AutoLR algorithm to determine the optimal initial learning rate.

#### Early Stopping Threshold ####

---
**yaml identifier: early_stop_threshold**

***Note that early stopping should only be used for the SGD with AdaS algorithm. As per the paper, AdaS provides the ability to monitor simply training loss and be confident that a low training loss leads to a high test accuracy/low testing loss. Hence, we only use early stop for SGD with AdaS, monitoring the training loss, and do not recommend its use otherwise. ***

The threshold for early stopping. The early stopping criterion operates by keeping track of the best loss seen to date, and evaluates the current loss against the best loss by doing `current_loss - best_loss`. If this value is **greater than** the early stopping threshold, a counter begins. If this evaluation is true for `early_stop_patience` (see below) amount of epochs, then early stopping is activated.

To deactivate early_stopping, set this value to `-1`.

#### Optimizer Arguments ####

---
**yaml identifier: optimizer_kwargs**

Specific arguments to pass to the selected optimizer. Expecting a dictionary, where keys are the exact argument names. There are certain required arguments for certain optimizers, which can be seen listed in [src/autolr/optim/__init__.py](src/autolr/optim/__init__.py). If not passing any arguments, ensure an empty list is the value.

#### Scheduler Arguments ####

---
**yaml identifier: scheduler_kwargs**

Same as above, but for scheduler argument.

#### Number of Training Trials ####

---
**yaml identifier: n_trials**

Number of full training cycles


#### Max Epochs ####

---
**yaml identifier: max_epoch**

Maximum number of epochs for one trial

#### Mini-Batch Size ####

---
**yaml identifier: mini_batch_size**

Size of mini-batch for one epoch

#### Early Stopping Patience ####

---
**yaml identifier: early_stop_patience**

Patience window for early stopping.

#### Power ####

---
**yaml identifier: p**

Power value for computing knowledge-gain. Can either be `1` or `2`.


#### Training Outputs ####
##### XLSX Output #####
Note that training progress is conveyed through console outputs, where per-epoch statements are outputed to indicate epoch time and train/test loss/accuracy.

Note also that a per-epoch `.csv` file is created for every training session. This file reports performance metrics of the CNN during training.
Table 2: Performance metrics of VGG16 trained on CIFAR100 - 1 Epoch
| Conv Block  | Train_loss_epoch_0 | in_S_epoch_0      | out_S_epoch_0     | fc_S_epoch_0      | in_rank_epoch_0 | out_rank_epoch_0 | fc_rank_epoch_0 | in_condition_epoch_0 | out_condition_epoch_0 | rank_velocity_0 | learning_rate_0 | acc_epoch_0 |
|----|--------------------|-------------------|-------------------|-------------------|-----------------|------------------|-----------------|----------------------|-----------------------|-----------------|-----------------|-------------|
| 0  | 4.52598691779329   | 0                 | 0                 | 0.007025059778243 | 0               | 0                | 0.01953125      | 0                    | 0                     | 0.03            | 0.03            | 0.0353      |
| 1  | 4.52598691779329   | 0.06363195180893  | 0.067244701087475 | 0.007025059778243 | 0.125           | 0.15625          | 0.01953125      | 4.14393854141235     | 5.25829029083252      | 0.03            | 0.03            | 0.0353      |
| 2  | 4.52598691779329   | 0.062127389013767 | 0.030436672270298 | 0.007025059778243 | 0.109375        | 0.046875         | 0.01953125      | 3.57764577865601     | 2.39811992645264      | 0.03            | 0.03            | 0.0353      |
| 3  | 4.52598691779329   | 0.035973243415356 | 0.030653497204185 | 0.007025059778243 | 0.0703125       | 0.0546875        | 0.01953125      | 3.60598373413086     | 3.2860517501831       | 0.03            | 0.03            | 0.0353      |
| 4  | 4.52598691779329   | 0.021210107952356 | 0.014563170261681 | 0.007025059778243 | 0.0390625       | 0.01953125       | 0.01953125      | 3.49767923355102     | 1.73739552497864      | 0.03            | 0.03            | 0.0353      |
| 5  | 4.52598691779329   | 0.017496244981885 | 0.018149495124817 | 0.007025059778243 | 0.03125         | 0.03125          | 0.01953125      | 3.05637526512146     | 2.64313006401062      | 0.03            | 0.03            | 0.0353      |
| 6  | 4.52598691779329   | 0.011354953050613 | 0.010315389372408 | 0.007025059778243 | 0.01953125      | 0.015625         | 0.01953125      | 2.54586839675903     | 2.25333142280579      | 0.03            | 0.03            | 0.0353      |
| 7  | 4.52598691779329   | 0.006322608795017 | 0.006018768996    | 0.007025059778243 | 0.01171875      | 0.0078125        | 0.01953125      | 3.68418765068054     | 2.13097596168518      | 0.03            | 0.03            | 0.0353      |
| 8  | 4.52598691779329   | 0.006788529921323 | 0.009726315736771 | 0.007025059778243 | 0.013671875     | 0.015625         | 0.01953125      | 3.65298628807068     | 2.70360684394836      | 0.03            | 0.03            | 0.0353      |
| 9  | 4.52598691779329   | 0.006502093747258 | 0.008573451079428 | 0.007025059778243 | 0.013671875     | 0.013671875      | 0.01953125      | 3.25959372520447     | 2.38875222206116      | 0.03            | 0.03            | 0.0353      |
| 10 | 4.52598691779329   | 0.003374363761395 | 0.005663644522429 | 0.007025059778243 | 0.0078125       | 0.0078125        | 0.01953125      | 4.67283821105957     | 2.17876362800598      | 0.03            | 0.03            | 0.0353      |
| 11 | 4.52598691779329   | 0.00713284034282  | 0.007544621825218 | 0.007025059778243 | 0.013671875     | 0.01171875       | 0.01953125      | 3.79078459739685     | 3.62017202377319      | 0.03            | 0.03            | 0.0353      |
| 12 | 4.52598691779329   | 0.006892844568938 | 0.007025059778243 | 0.007025059778243 | 0.017578125     | 0.01953125       | 0.01953125      | 6.96407127380371     | 8.45268821716309      | 0.03            | 0.03            | 0.0353      |

Where each row represents a single convolutional block and:
- **Train_loss_epoch_0** is the training loss for 0-th epoch
- **in_S_epoch_0** is the knowledge gain for the input to that conv block for 0-th epoch
- **out_S_epoch_0** is the knowledge gain for the output of that conv block for 0-th epoch
- **fc_S_epoch_0** is the knowledge gain for the fc portion of that conv block for 0-th epoch
- **in_rank_epoch_0** is the rank for the input to that conv block for 0-th epoch
- **out_rank_epoch_0** is the rank for the output of that conv block for 0-th epoch
- **fc_rank_epoch_0** is the rank for the fc portion of that conv block for 0-th epoch
- **in_condition_epoch_0** is the mapping condition for the input to that conv block for 0-th epoch
- **out_condition_epoch_0** is the mapping condition for the output of that conv block for 0-th epoch
- **rank_velocity_epoch_0** is the rank velocity for that conv block for the 0-th epoch
- **learning_rate_epoch_0** is the learning rate for that conv block for all parameters for the 0-th epoch
- **acc_epoch_0** is the testing accuracy for the 0-th epoch

The columns will continue to grow during training, appending each epoch's metrics each time.

The location of the output `.csv` file depends on the `--root` and `--output` option during training, and naming of the file is determined by the `config.yaml` file's contents.

##### Checkpoints #####
Checkpoints are saved to the path specified by the `-root` and `--checkpoint` option. A file or directory may be passed. If a directory path is specified, the filename for the checkpoint defaults to `ckpt.pth`.

### Common Issues (running list) ###
- NONE :)

## TODO ###
- Extension of AdaS to Deep Neural Networks

### Pytest ###
Note the following:
- Our Pytests write/download data/files etc. to `/tmp`, so if you don't have a `/tmp` folder (i.e. you're on Windows), then correct this if you wish to run the tests yourself
