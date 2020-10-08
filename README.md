# autoHyper: Response Modeling of Hyper-Parameters for Deep Convolutional Neural Networks#
[Paper]()
## Status ##
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![maintenance](https://img.shields.io/badge/maintained%3F-yes-brightgreen.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![python](https://img.shields.io/badge/python-v3.7-blue)](https://www.python.org/downloads/release/python-370/)
![size](https://img.shields.io/github/repo-size/MathieuTuli/AutoLR)

## Table of Contents ##

### Introduction ###
[autoHyper]() is an algorithm to automatically determine the optimal initial learning rate for Neural Networks:
- it exhibits rapid convergence (on the order of minutes and hours)
- it generalizes well to model, dataset, optimizer selection (amoungst other experimental settings)

This repository contains a [PyTorch](https://pytorch.org/) implementation of autoHyper.

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
### Requirements ###
#### Software/Hardware ####
We use `Python 3.7` (although compatibility with versions >= 3.7 *should not* pose an issue)

Refer [requirements.txt](requirements.txt) for the required Python Packages. Additional details can be found on the [Requirements Wiki]()

### Installation ###
There are two versions of the AdaS code contained in this repository.
1. a python-package version of the AdaS code, which can be `pip`-installed.
2. a static python module (unpackaged), runable as a script.

All source code can be found in [src/autohyper](src/autohyper). Additional details can be found on the [Installation Wiki]()

### Usage ###
Moving forward, I will refer to console usage of this library. IDE usage is no different. Training options are split two ways:
1. First, all environment/infrastructure options (GPU usage, output paths, etc.) is specified using arguments.
2. Second, all training specific options (network, dataset, hyper-parameters, etc.) is specified using a [YAML](https://yaml.org/) configuration file.

#### Training Outputs ####
In addition to console outputs, all information is also logged in `csv` files during training.

### Common Issues (running list) ###
- NONE :)

## TODO ###
- Extension of AdaS to Deep Neural Networks

### Pytest ###
Note the following:
- Our Pytests write/download data/files etc. to `/tmp`, so if you don't have a `/tmp` folder (i.e. you're on Windows), then correct this if you wish to run the tests yourself
