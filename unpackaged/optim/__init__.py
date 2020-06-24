"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Any

import torch

from optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, OneCycleLR
from optim.novograd import NovoGrad
from optim.adabound import AdaBound
from optim.adashift import AdaShift
from optim.adadelta import Adadelta
from optim.adagrad import Adagrad
from optim.rmsprop import RMSprop
from optim.nosadam import NosAdam
from optim.laprop import LaProp
from optim.adamod import AdaMod
from optim.adamax import Adamax
from optim.nadam import NAdam
from optim.padam import PAdam
from optim.radam import RAdam
from optim.sgd import SGDVec
from optim.adam import Adam
from optim.sgd import SGD
from optim.sps import SPS
from optim.sls import SLS
from optim.lrd import LRD


def get_optimizer_scheduler(net_parameters: Any,
                            init_lr: float, optim_method: str,
                            lr_scheduler: str,
                            train_loader_len: int,
                            max_epochs: int) -> torch.nn.Module:
    optimizer = None
    scheduler = None
    if optim_method == 'SGD':
        if lr_scheduler == 'AdaS':
            optimizer = SGDVec(
                net_parameters, lr=init_lr,
                momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = SGD(
                net_parameters, lr=init_lr,
                momentum=0.9, weight_decay=5e-4)
    elif optim_method == 'NAG':
        optimizer = SGD(
            net_parameters, lr=init_lr,
            momentum=0.9, weight_decay=5e-4,
            nesterov=True)
    elif optim_method == 'AdaM':
        optimizer = Adam(net_parameters, lr=init_lr)
    elif optim_method == 'AdaGrad':
        optimizer = Adagrad(net_parameters, lr=init_lr)
    elif optim_method == 'RMSProp':
        optimizer = RMSprop(net_parameters, lr=init_lr)
    elif optim_method == 'AdaDelta':
        optimizer = Adadelta(net_parameters, lr=init_lr)
    elif optim_method == 'AdaBound':
        optimizer = AdaBound(net_parameters, lr=init_lr)
    # below = untested
    elif optim_method == 'AdaMax':
        optimizer = Adamax(net_parameters, lr=init_lr)
    elif optim_method == 'AdaMod':
        optimizer = AdaMod(net_parameters, lr=init_lr)
    elif optim_method == 'AdaShift':
        optimizer = AdaShift(net_parameters, lr=init_lr)
    elif optim_method == 'NAdam':
        optimizer = NAdam(net_parameters, lr=init_lr)
    elif optim_method == 'NosAdam':
        optimizer = NosAdam(net_parameters, lr=init_lr)
    elif optim_method == 'NovoGrad':
        optimizer = NovoGrad(net_parameters, lr=init_lr)
    elif optim_method == 'PAdam':
        optimizer = PAdam(net_parameters, lr=init_lr)
    elif optim_method == 'RAdam':
        optimizer = RAdam(net_parameters, lr=init_lr)
    elif optim_method == 'SPS':
        optimizer = SPS(net_parameters, init_step_size=init_lr)
    elif optim_method == 'SLS':
        optimizer = SLS(net_parameters, init_step_size=init_lr)
    elif optim_method == 'LaProp':
        optimizer = LaProp(net_parameters, lr=init_lr)
    elif optim_method == 'LearningRateDropout':
        optimizer = LRD(net_parameters, lr=init_lr,
                        lr_dropout_rate=0.5)
    else:
        print(f"Adas: Warning: Unknown optimizer {optim_method}")
    if lr_scheduler == 'StepLR':
        scheduler = StepLR(
            optimizer, step_size=25, gamma=0.5)
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        first_restart_epochs = 25
        increasing_factor = 1
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=first_restart_epochs, T_mult=increasing_factor)
    elif lr_scheduler == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer, max_lr=init_lr,
            steps_per_epoch=train_loader_len, epochs=max_epochs)
    elif lr_scheduler != 'AdaS':
        print(f"Adas: Warning: Unknown LR scheduler {lr_scheduler}")
    return (optimizer, scheduler)
