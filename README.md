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
There are a few versions to consider:
1. The algorithm itself can be found in the `main` branch of this repository
2. The `single-hp-bayesian` and `multi-hp` branches provide code for experiments in our paper. The `multi-hp` branch is the latest branch for the current paper, while `single-hp-bayesian` has an earlier version, that shows another method for single-hp optimization.

All source code can be found in [src/autohyper](src/autohyper).

To install the autoHyper algorithm then, you can simply run `pip install autohyper` or clone this repository and run `pip install -e .`.

### Usage ###
For the packaged code, after installation, training can be run using the following command: `python -m autohyper train ...`

For the unpackaged code, training can be run using the following command: `python train.py ...` ([src/autohyper/train.py](src/autohyper/train.py))

To use autoHyper is quite simpler and follow similar HPO packages. See below for a skeleton of what needs to happen:

```python
def main():
    hyper_parameters = HyperParameters(lr=True, weight_decay=True)
```
