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
from scipy.stats import loguniform
from typing import Union, List
from datetime import datetime
import sys

# import logging

import numpy as np

mod_name = vars(sys.modules[__name__])['__package__']

if mod_name:
    from .metrics import Metrics
else:
    from metrics import Metrics


def compute_metric(metrics: Metrics) -> float:
    return np.mean([np.mean(np.isclose(
        metric.input_channel_S + metric.output_channel_S,
        0).astype(np.float16)) for
        metric in metrics.historical_metrics])


def compute_rate_of_change(rank_history: List[float]) -> float:
    if len(rank_history) == 1:
        return [1.]
    cumprod = np.cumprod(rank_history)
    output = list()
    for i in range(len(rank_history)):
        temp = cumprod[:i+1] - (rank_history[i] * (i+1)) + 1e-7
        if i == 0:
            output.append(1.0)
        else:
            temp = (temp[i] - temp[i-1])/temp[i]
            output.append(temp)
    return output


def auto_lr(training_agent,
            min_lr: float = 1e-4,
            max_lr: float = 0.1,
            num_split: int = 25,
            min_delta: float = 5e-3,
            lr_delta: float = 3e-5,
            epochs: Union[range, List[int]] = range(0, 5),
            exit_counter_thresh: int = 6,
            power: float = 0.8):
    # ###### LR RANGE STUFF #######
    learning_rates = np.geomspace(min_lr, max_lr, num_split)
    auto_lr_path = training_agent.output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    grid = dict()
    if training_agent.config['dataset'] == 'TinyImageNet':
        if training_agent.config['optimizer'] == 'AdaM':
            r = 15
        elif training_agent.config['optimizer'] == 'AdaBound':
            r = 30
        elif training_agent.config['optimizer'] == 'AdaGrad':
            r = 32
        elif training_agent.config['scheduler'] == 'AdaS':
            r = 40
    elif training_agent.config['dataset'] == 'CIFAR10':
        if training_agent.config['optimizer'] == 'AdaM':
            r = 15
        elif training_agent.config['optimizer'] == 'AdaBound':
            r = 10
        elif training_agent.config['optimizer'] == 'AdaGrad':
            r = 25
        elif training_agent.config['scheduler'] == 'AdaS':
            r = 34
    if training_agent.config['network'] == 'DenseNet121CIFAR10':
        if training_agent.config['optimizer'] == 'AdaM':
            r = 19
        elif training_agent.config['optimizer'] == 'AdaBound':
            r = 24
        elif training_agent.config['optimizer'] == 'AdaGrad':
            r = 32
        elif training_agent.config['scheduler'] == 'AdaS':
            r = 36
    trials = {
        'ResNet18CIFAR': {
            'CIFAR10': {
                'AdaM': 10,
                'AdaBound': 12,
                'AdaGrad': 20,
                'SGD': 28},
            'CIFAR100': {
                'AdaM': 16,
                'AdaBound': 17,
                'AdaGrad': 27,
                'SGD': 28}
        },
        'ResNeXtCIFAR': {
            'CIFAR10': {
                'AdaM': 17,
                'AdaBound': 20,
                'AdaGrad': 32,
                'SGD': 33},
            'CIFAR100': {
                'AdaM': 17,
                'AdaBound': 19,
                'AdaGrad': 33,
                'SGD': 35}
        },
        'DenseNet121CIFAR': {
            'CIFAR10': {
                'AdaM': 19,
                'AdaBound': 24,
                'AdaGrad': 32,
                'SGD': 36},
            'CIFAR100': {
                'AdaM': 19,
                'AdaBound': 24,
                'AdaGrad': 33,
                'SGD': 37}
        },
        'ResNet34CIFAR': {
            'CIFAR10': {
                'AdaM': 10,  # 10 for 1d, 20 for 2d
                'AdaBound': 11,
                'AdaGrad': 20,  # 20 for 1d, 25 for 2d
                'SGD': 27},
            'CIFAR100': {
                'AdaM': 10,
                'AdaBound': 13,
                'AdaGrad': 32,
                'SGD': 26},
            'TinyImageNet': {
                'AdaM': 14,
                'AdaBound': 15,
                'AdaGrad': 22,
                'SGD': 30}
        }
    }
    r = trials[training_agent.config['network']
               ][training_agent.config['dataset']
                 ][training_agent.config['optimizer']]
    for i in range(r):
        learning_rate = loguniform.rvs(min_lr, max_lr)
        training_agent.reset(learning_rate)
        training_agent.output_filename = training_agent.output_path / 'auto-lr'
        training_agent.output_filename.mkdir(exist_ok=True, parents=True)
        string_name = \
            "auto_lr_results_" +\
            f"date={date}_" +\
            f"network={training_agent.config['network']}_" +\
            f"dataset={training_agent.config['dataset']}_" +\
            f"optimizer={training_agent.config['optimizer']}_" +\
            '_'.join([f"{k}={v}" for k, v in
                      training_agent.config['optimizer_kwargs'].items()]) +\
            f"scheduler={training_agent.config['scheduler']}_" +\
            '_'.join([f"{k}={v}" for k, v in
                      training_agent.config['scheduler_kwargs'].items()]) +\
            f"learning_rate={learning_rate}" +\
            ".csv".replace(' ', '-')
        training_agent.output_filename = str(
            training_agent.output_filename / string_name)
        training_agent.run_epochs(trial=0, epochs=epochs)

        # ###### LR RANGE STUFF #######
        if training_agent.config['metric'] == 'loss':
            grid[learning_rate] = \
                training_agent.performance_statistics['test_loss_epoch_4']
        else:
            grid[learning_rate] = \
                training_agent.performance_statistics['test_acc1_epoch_4']
        print(
            f"Learning rate {learning_rate} acc1 {grid[learning_rate]}")

    with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
        f.write('lr,acc1\n')
        for lr, acc in grid.items():
            f.write(f'{lr},{acc},\n')

    if training_agent.config['metric'] == 'loss':
        return min(grid, key=grid.get)
    else:
        return max(grid, key=grid.get)
