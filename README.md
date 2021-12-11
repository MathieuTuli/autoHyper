# Towards Robust and Automatic Hyper-Parameter Tuning #
**Good News!: the autoHyper work (Towards Robust and Automatic Hyper-Parameter Tuning) has been accepted to the OPT2021 Neurips Workshop**

The paper can be accessed here:
> [**Towards Robust and Automatic Hyper-Parameter Tuning**](https://arxiv.org/abs/2111.14056)   

**autoHyper** is a hyper-parameter optimization (HPO) method that monitors the low-rank factorized weights of intermediate layers in a neural network in order to develop a surrogate metric to model validation performance.
## Status ##
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![maintenance](https://img.shields.io/badge/maintained%3F-yes-brightgreen.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![python](https://img.shields.io/badge/python-v3.7-blue)](https://www.python.org/downloads/release/python-370/)

## Table of Contents ##
* [Citing autoHyper](#citing-autohyper)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Training Outputs](#training-outputs)
* [TODO](#todo)
* [Pytest](#pytest)

### Citing autoHyper ###
```text
@misc{tuli2021robust,
      title={Towards Robust and Automatic Hyper-Parameter Tunning}, 
      author={Mathieu Tuli and Mahdi S. Hosseini and Konstantinos N. Plataniotis},
      year={2021},
      eprint={2111.14056},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

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

Refer to the [Usage Wiki](Usage.md) for additional details.

### Training Outputs ###
In addition to console outputs, all information is also logged in `csv` files during training.  Details can be foudn in the [Outputs Wiki](Outputs.md)

### Pytest ###
Note the following:
- Our Pytests write/download data/files etc. to `/tmp`, so if you don't have a `/tmp` folder (i.e. you're on Windows), then correct this if you wish to run the tests yourself
