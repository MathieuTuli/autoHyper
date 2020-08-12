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
  * [Software](#software)
  * [Hardware](#hardware)
  * [Some Experimental Results](#some-experimental-results)
- [Installation](#installation)
  * [Python Package](#python-package)
    + [Direct pip-installation](#direct-pip-installation)
    + [Repository Cloning](#repository-cloning)
  * [Unpackaged Python](#unpackaged-python)
- [Usage](#usage)
  * [Training](#training)
  * [Config Options](#config-options)
    + [Available Datasets for Training](#available-datasets-for-training)
    + [Available Models for Training](#available-models-for-training)
    + [Optimizer Method](#optimizer-method)
    + [Learning Rate Scheduler](#learning-rate-scheduler)
    + [Number of Training Trials](#number-of-training-trials)
    + [Beta](#beta)
    + [Initial Learning Rate](#initial-learning-rate)
    + [Max Epochs](#max-epochs)
    + [Early Stopping Threshold](#early-stopping-threshold)
    + [Early Stopping Patience](#early-stopping-patience)
    + [Mini-Batch Size](#mini-batch-size)
    + [Minimum Learning Rate](#minimum-learning-rate)
    + [Zeta](#zeta)
    + [Power](#power)
  * [Training Outputs](#training-outputs)
    + [XLSX Output](#xlsx-output)
    + [Checkpoints](#checkpoints)
- [Common Issues (running list)](#common-issues--running-list-)
- [Pytest](#pytest)

## Introduction ##
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
```

**NOTE** that in order to satisfy `torch==1.5.0` the following Nvidia requirements need to be met:
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
There are two version of the AdaS code contained in this repository.
1. a python-package version of the AdaS code, which can be `pip`-installed.
2. a static python module, provided in addition to the package for any user who may want an unpackaged version of the code. This code can be found at [unpackaged](unpackaged)

#### Python Package ####

---

##### Repository Cloning #####
After cloning the repository, simply run
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

#### Unpackaged Python ####

---
Ensure first that you have the requirements installed per [requirements.txt](requirements.txt)

You can run the code either by typing
```console
python main.py train --...
```
OR
```console
python train.py --...
```
Where `--...` represents the options for training (see below)

### Usage ###
#### Training ####
As this is a self-contained python module, all functionalities are built in once you `pip`-install the package. To start training, you must first define you configuration file. An example can be found at [src/adas/config.yaml](src/adas/config.yaml). Finally, run
```console
python -m adas train --config *config.yaml*
```
Where you specify the path of you `config.yaml` file. Note the following options for adas:
```console

python -m adas train --help
--
usage: __main__.py train [-h] [--config CONFIG] [--data DATA]
                         [--output OUTPUT] [--checkpoint CHECKPOINT]
                         [--root ROOT] [-r]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Set configuration file path: Default = 'config.yaml'
  --data DATA           Set data directory path: Default = '.adas-data'
  --output OUTPUT       Set output directory path: Default = '.adas-output'
  --checkpoint CHECKPOINT
                        Set checkpoint directory path: Default = '.adas-checkpoint
  --root ROOT           Set root path of project that parents all others:
                        Default = '.'
  -r, --resume          Flag: resume training from checkpoint
```
#### Config Options ####
In the following sections we list the configuration options available to the user. Note that we also classify the configuration options into the following categories:
- Suggested Default
  - max_epoch
  - early_stop_threshold
  - early_stop_patience
  - min_lr
  - zeta
  - p
  - loss
- Suggested Tune
  - n_trials
  - beta
  - init_lr
- Application Specific
  - dataset
  - network
  - optim_method
  - lr_scheduler

The **Suggested Default** parameters are ones we have preset and suggest not be altered too much. Naturally, the user may change them at their discretion, however these values were found to be stable and optimal.

The **Suggested Tune** parameters are highly recommended to be tuned, and are very application specific.

The **Application Specific** parameters then are simply ones that the user must change to do what they want (what dataset, model, learning algorithm, etc.)


##### Available Datasets for Training #####
---
**yaml identifier: dataset**
Currently only the following datasets are supported:
- CIFAR10
- CIFAR100
- ImageNet (see [Common Issues](#common-issues--running-list-))

##### Available Models for Training #####

---
**yaml identifier: network**
All models used can be found in [src/adas/models](src/adas/models) in this repository are copied from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). We note that modifications were made to these models to allow variable `num_classes` to be used, relative to the dataset being used for training. The available models are as follows:
- VGG16
- ResNet34
- PreActResNet18
- GoogLeNet
- densenet_cifar
- ResNeXt29_2x64d
- MobileNet
- MobileNetV2
- DPN92
- ShuffleNetG2
- SENet18
- ShuffleNetV2
- EfficientNetB0

##### Optimizer Method #####

---
**yaml identifier: optim_method**

Options:
- SGD
- AdaM
- AdaGrad
- RMSProp
- AdaDelta

##### Learning Rate Scheduler #####

---
**yaml identifier: lr_scheduler**

Options:
- AdaS (Note that `SGD` must be specified as the `optim_method`)
- StepLR
- CosineAnnealingWarmRestarts
- OneCycleLR

##### Number of Training Trials #####

---
**yaml identifier: n_trials**

Number of full training cycles

##### Beta #####

---
**yaml identifier: beta**

AdaS gain factor. Tunes the AdaS behaviour. Smaller means faster convergence, but lower final testing loss, and vice-versa.

##### Initial Learning Rate #####

---
**yaml identifier: init_lr**

Initial learning rate for the optimizer method

##### Max Epochs #####

---
**yaml identifier: max_epoch**

Maximum number of epochs for one trial

##### Early Stopping Threshold #####

---
**yaml identifier: early_stop_threshold**

***Note that early stopping should only be used for the SGD with AdaS algorithm. As per the paper, AdaS provides the ability to monitor simply training loss and be confident that a low training loss leads to a high test accuracy/low testing loss. Hence, we only use early stop for SGD with AdaS, monitoring the training loss, and do not recommend its use otherwise. ***

The threshold for early stopping. The early stopping criterion operates by keeping track of the best loss seen to date, and evaluates the current loss against the best loss by doing `current_loss - best_loss`. If this value is **greater than** the early stopping threshold, a counter begins. If this evaluation is true for `early_stop_patience` (see below) amount of epochs, then early stopping is activated.

To deactivate early_stopping, set this value to `-1`.

##### Early Stopping Patience #####

---
**yaml identifier: early_stop_patience**

Patience window for early stopping.

##### Mini-Batch Size #####

---
**yaml identifier: mini_batch_size**

Size of mini-batch for one epoch

##### Minimum Learning Rate #####

---
**yaml identifier: mini_batch_size**

Size of mini-batch for one epoch

##### Zeta #####

---
**yaml identifier: zeta**

The knowledge-gain hyper-parameter, another AdaS hyper-parameter. Typically always set to 1.

##### Power #####

---
**yaml identifier: p**

Power value for computing knowledge-gain. Can either be `1` or `2`.


#### Training Outputs ####
##### XLSX Output #####
Note that training progress is conveyed through console outputs, where per-epoch statements are outputed to indicate epoch time and train/test loss/accuracy.

Note also that a per-epoch updating `.xlsx` file is created for every training session. This file reports performance metrics of the CNN during training. An example is show in Table 2.
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

The location of the output `.xlsx` file depends on the `-root` and `--output` option during training, and naming of the file is determined by the `config.yaml` file's contents.

##### Checkpoints #####
Checkpoints are saved to the path specified by the `-root` and `--checkpoint` option. A file or directory may be passed. If a directory path is specified, the filename for the checkpoint defaults to `ckpt.pth`.

### Common Issues (running list) ###
- NONE :)

## TODO ###
- Extension of AdaS to Deep Neural Networks

### Pytest ###
Note the following:
- Our Pytests write/download data/files etc. to `/tmp`, so if you don't have a `/tmp` folder (i.e. you're on Windows), then correct this if you wish to run the tests yourself
