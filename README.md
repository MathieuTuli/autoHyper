# autoHyper: Response Modeling of Hyper-Parameters for Deep Convolutional Neural Networks #
[Paper](https://arxiv.org/pdf/2111.14056.pdf)
## Status ##
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![maintenance](https://img.shields.io/badge/maintained%3F-yes-brightgreen.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![python](https://img.shields.io/badge/python-v3.7-blue)](https://www.python.org/downloads/release/python-370/)
![size](https://img.shields.io/github/repo-size/autoHyper)

## Table of Contents ##
* [License](#license)
* [Citing autoHyper](#citing-autohyper)
* [Introduction](#introduction)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Training Outputs](#training-outputs)
* [Common Issues (running list)](#common-issues--running-list-)
* [TODO](#todo)
* [Pytest](#pytest)

### License ###
autoHyper is released under the MIT License (refer to the [LICENSE](LICENSE) file for more information)
|Permissions|Conditions|Limitations|
|---|---|---|
|![license](https://img.shields.io/badge/-%20-brightgreen) Commerical use|![license](https://img.shields.io/badge/-%20-blue) License and Copyright Notice|![license](https://img.shields.io/badge/-%20-red) Liability|
|![license](https://img.shields.io/badge/-%20-brightgreen) Distribution| | ![license](https://img.shields.io/badge/-%20-red) Warranty|
|![license](https://img.shields.io/badge/-%20-brightgreen) Modification | | |
|![license](https://img.shields.io/badge/-%20-brightgreen) Private Use| | |

### Citing autoHyper ###
```text
@misc{
}
```

### Introduction ###
[autoHyper]() is an algorithm that automatically determines the optimal initial learning rate for Neural Networks without the need to perform time consuming grid searching techniques:
- it exhibits rapid convergence (on the order of minutes and hours)
- it generalizes well to model, dataset, optimizer selection (amoungst other experimental settings)
- it always achieves competitive performance (<1% difference on top-1 testing accuracy) and in some cases, drasticaly improves (such as a 4.93% increase in top-1 testing accuracy for ResNet34 trainined using AdaM applied on ImageNet)

This repository contains a [PyTorch](https://pytorch.org/) implementation of autoHyper.

A comprehensive list of expected results can be found in the paper.

### Requirements ###
#### Software/Hardware ####
We use `Python 3.7` (although compatibility with versions >= 3.7 *should not* pose an issue)
```console
pip install -r requirements.txt
```
Additional details can be found on the [Requirements Wiki](Requirements.md)

### Installation ###
There are two versions of the AdaS code contained in this repository.
1. a python-package version of the AdaS code, which can be `pip`-installed.
 - `pip install -e .` or `pip install .` will install the package
2. a static python module (unpackaged), runable as a script.

All source code can be found in [src/autohyper](src/autohyper).

### Usage ###
Moving forward, I will refer to console usage of this library. IDE usage is no different. Training options are split two ways:
1. First, all environment/infrastructure options (GPU usage, output paths, etc.) is specified using arguments.
2. Second, all training specific options (network, dataset, hyper-parameters, etc.) is specified using a [YAML](https://yaml.org/) configuration file.

For the packaged code, after installation, training can be run using the following command: `python -m autohyper train ...`

For the unpackaged code, training can be run using the following command: `python train.py ...` ([src/autohyper/train.py](src/autohyper/train.py))


```console
python -m autohyper train --help

usage: __main__.py train [-h] [--config CONFIG] [--data DATA]
                         [--output OUTPUT] [--checkpoint CHECKPOINT]
                         [--resume RESUME] [--root ROOT]
                         [--save-freq SAVE_FREQ] [--cpu] [--gpu GPU]
                         [--multiprocessing-distributed] [--dist-url DIST_URL]
                         [--dist-backend DIST_BACKEND]
                         [--world-size WORLD_SIZE] [--rank RANK]
```
The following is an example config file:
```yaml
###### Application Specific ######
dataset: 'CIFAR10'
network: 'ResNet18CIFAR'
optimizer: 'AdaM'
scheduler: 'None'

###### Suggested Tune ######
init_lr: 0.01
early_stop_threshold: 0.001
optimizer_kwargs: {}
scheduler_kwargs: {}


###### Suggested Default ######
n_trials: 5
num_workers: 4
max_epochs: 100
loss: 'cross_entropy'
mini_batch_size: 128
early_stop_patience: 10
p: 1
```

Refer to the [Usage Wiki](Usage.md) for additional details.

### Training Outputs ###
In addition to console outputs, all information is also logged in `csv` files during training.  Details can be foudn in the [Outputs Wiki](Outputs.md)

### Common Issues (running list) ###
- NONE :)

### TODO ###
- Extension of AdaS to Deep Neural Networks

### Pytest ###
Note the following:
- Our Pytests write/download data/files etc. to `/tmp`, so if you don't have a `/tmp` folder (i.e. you're on Windows), then correct this if you wish to run the tests yourself
