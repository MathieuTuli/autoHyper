"""
MIT License

Copyright (c) 2020

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
from typing import Union, List, Dict, Any, Tuple
from datetime import datetime
import sys

# import logging

import numpy as np

mod_name = vars(sys.modules[__name__])['__package__']

if mod_name:
    from .metrics import Metrics
    from .components import HyperParameters
else:
    from metrics import Metrics
    from components import HyperParameters


def compute_rank(metrics: Metrics) -> float:
    per_S_zero = np.mean([np.mean(np.isclose(
        metric.input_channel_S + metric.output_channel_S,
        0).astype(np.float16)) for
        metric in metrics.historical_metrics])
    return per_S_zero


def auto_lr(training_agent,
            hyper_parameters: HyperParameters,
            num_split: int = 20,
            min_delta: float = 5e-3,
            # lr_delta: float = 3e-5,
            epochs: Union[range, List[int]] = range(0, 5),
            power: float = 0.8):
    # learning_rates = np.geomspace(min_lr, max_lr, num_split)
    config = training_agent.config.deepcopy()
    delta = min_delta
    first_run = True
    output_history = list()
    rank_history = list()
    cur_rank = -1
    param = 'init_lr'
    auto_lr_path = training_agent.output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    buffer = list()
    while True:
        with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
            f.write('lr,rank,msg\n')
            for (lr, rank, msg) in output_history:
                f.write(f'{lr},{rank},{msg}\n')
        training_agent.reset(config)
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
            f"learning_rate={config['init_lr']}" +\
            ".csv".replace(' ', '-')
        training_agent.output_filename = str(
            training_agent.output_filename / string_name)
        training_agent.run_epochs(trial=0, epochs=epochs)

        # ###### LR RANGE STUFF #######
        cur_rank = compute_rank(training_agent.metrics)
        rank_history.append(cur_rank)
        rate_of_change = np.cumprod(rank_history) ** power
        # print(f"LR Range Test: Cur. Grid Space: {learning_rates}")
        # print(f"LR Range Test: Cur. LR: {learning_rates[lr_idx]}")
        # print(f"LR Range Test: Cur. Rank: {cur_rank}")
        if first_run and np.less(cur_rank, 0.5):
            hyper_parameters.config['init_lr'] /= 10
            continue
        else:
            first_run = False
            if np.isclose(cur_rank, 0.0):
                hyper_parameters.config[param]['current'] -= 2 * \
                    hyper_parameters.config[param]['step']
            if np.less(rate_of_change[-1], delta):
                hyper_parameters.config[param]['step'] /= 2
                delta = max(delta / 2, min_delta)
                continue
    with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
        f.write('lr,rank,msg\n')
        for (lr, rank, msg) in output_history:
            f.write(f'{lr},{rank},{msg}\n')
    return hyper_parameters
