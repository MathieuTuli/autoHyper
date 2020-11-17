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
    for i, learning_rate in enumerate(learning_rates):
        if training_agent.config['dataset'] == 'CIFAR100':
            if i > 9:
                break
        else:
            if i > 12:
                break
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
        grid[learning_rate] = \
            training_agent.performance_statistics['test_acc1_epoch_4']
        print(
            f"Learning rate {learning_rate} acc1 {grid[learning_rate]}")

    with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
        f.write('lr,acc1\n')
        for lr, acc in grid.items():
            f.write(f'{lr},{acc},\n')

    return max(grid, key=grid.get)
