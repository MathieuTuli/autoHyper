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
            scale_delta: float = 3e-5,
            epochs: Union[range, List[int]] = range(0, 5),
            power: float = 0.8):
    # learning_rates = np.geomspace(min_lr, max_lr, num_split)
    config = training_agent.config
    first_run = True
    output_history = list()
    cur_rank = -1
    auto_lr_path = training_agent.output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    delta = dict()
    for param in hyper_parameters.config.keys():
        delta[param] = .1
        if param == 'init_lr':
            config['init_lr'] = hyper_parameters.config[param]['current']
        elif param == 'weight_decay':
            config['optimizer_kwargs']['weight_decay'] = \
                hyper_parameters.config[param]['current']
    while True:
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
        # print(f"LR Range Test: Cur. Grid Space: {learning_rates}")
        # print(f"LR Range Test: Cur. LR: {learning_rates[lr_idx]}")
        # print(f"LR Range Test: Cur. Rank: {cur_rank}")
        for param in hyper_parameters.config.keys():
            if hyper_parameters.config[param]['stop']:
                continue
            if first_run and np.less(cur_rank, 0.5):
                print("FIRST RUN LESS THAN 50")
                hyper_parameters.config[param]['current'] /= 10
                continue
            else:
                first_run = False
                if np.isclose(cur_rank, 0.0):
                    print("RANK ZERO BACK UP")
                    hyper_parameters.config[param]['current'] *= \
                        hyper_parameters.config[param]['scale'] ** 2
                    continue
                hyper_parameters.config[param]['buffer'].append(cur_rank)
                zeta = np.cumprod(
                    hyper_parameters.config[param]['buffer']) ** power
                # hyper_parameters.config[param]['current'] *= \
                #     hyper_parameters.config[param]['scale']
                # if all(np.less(zeta[-3:], delta[param])) and len(zeta) > 2:
                if np.less(zeta[-1], min_delta):
                    print("LESS THAN DELTA")
                    if np.isclose(hyper_parameters.config[param]['scale'], 1.,
                                  atol=scale_delta):
                        hyper_parameters.config[param]['stop'] = True
                    hyper_parameters.config[param]['count'] += 1
                    hyper_parameters.config[param]['scale'] = .5 * \
                        np.exp(-hyper_parameters.config[param]
                               ['count'] * 2) + 1
                    # delta[param] = max(delta[param] / 2, min_delta)
                    hyper_parameters.config[param]['buffer'].clear()
                    # continue
                    hyper_parameters.config[param]['current'] /= \
                        hyper_parameters.config[param]['scale'] ** 3

            hyper_parameters.config[param]['current'] *= \
                hyper_parameters.config[param]['scale']
            if param == 'init_lr':
                config['init_lr'] = hyper_parameters.config[param]['current']
            elif param == 'weight_decay':
                config['optimizer_kwargs']['weight_decay'] = \
                    hyper_parameters.config[param]['current']
        for param, conf in hyper_parameters.config.items():
            print(param)
            print(conf)
        if all([config['stop'] for _, config in hyper_parameters.config.items()]):
            break
    with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
        f.write('lr,rank,msg\n')
        for (lr, rank, msg) in output_history:
            f.write(f'{lr},{rank},{msg}\n')
    return hyper_parameters
