
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
from typing import Union, List
from datetime import datetime
import itertools
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
            epochs: Union[range, List[int]] = range(0, 1),
            power: float = 0.8):
    establish_start = True
    cur_rank = -1
    auto_lr_path = training_agent.output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    delta = dict()
    for param in hyper_parameters.config.keys():
        delta[param] = .1
        if param == 'init_lr':
            training_agent.config['init_lr'] = \
                hyper_parameters.config[param].current
        elif param == 'weight_decay':
            training_agent.config['optimizer_kwargs']['weight_decay'] = \
                hyper_parameters.config[param].current

    num_hp = len(hyper_parameters.config.keys())
    rank_history = list()
    scale_powers = [-1, 0, 1]  # defines the trust region
    scale_powers = [0, 1]  # defines the trust region
    trust_region = list(itertools.product(
        *[scale_powers for i in range(num_hp)]))
    trust_buffer = np.full([len(scale_powers)] * num_hp, -1., dtype=float)

    def reset():
        training_agent.reset(training_agent.config)
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
            f"learning_rate={training_agent.config['init_lr']}" +\
            ".csv".replace(' ', '-')
        training_agent.output_filename = str(
            training_agent.output_filename / string_name)
    # for 2 hyper-parameters and 3 hyper-parameters,
    # the buffer is a 3x3 matrix, a 3x3x3 cube, a ...

    while True:
        print('--')
        print("Current settings:")
        print(f"trust_buffer: {trust_buffer}")
        print(f"rank_history: {rank_history}")
        print(f"params: {hyper_parameters.config}")
        print('--')

        if establish_start:
            reset()
            training_agent.run_epochs(trial=0, epochs=epochs)
            cur_rank = compute_rank(training_agent.metrics)
            if np.less(cur_rank, .9):
                print("FIRST RUN LESS THAN 90%")
                print(f"Rank was {cur_rank}")
                for param in hyper_parameters.config.keys():
                    hyper_parameters.config[param].current /= \
                        hyper_parameters.config[param].scale ** 2
                    if param == 'init_lr':
                        training_agent.config['init_lr'] = \
                            hyper_parameters.config[param].current
                    elif param == 'weight_decay':
                        training_agent.config['optimizer_kwargs']['weight_decay'
                                                                  ] = \
                            hyper_parameters.config[param].current
                continue
        establish_start = False
        for scale_power in trust_region:
            print('--')
            print("Doing TR")
            print(f"params: {hyper_parameters.config}")
            print(f"trust_region: {trust_buffer}")
            print(f"params: {training_agent.config}")
            # index = tuple(np.array(scale_power) + 1)
            index = tuple(np.array(scale_power))
            if np.less(trust_buffer[index], 0.):
                for i, param in enumerate(hyper_parameters.config.keys()):
                    current = hyper_parameters.config[param].current * \
                        (hyper_parameters.config[param].scale **
                         scale_power[i])
                    if param == 'init_lr':
                        training_agent.config['init_lr'] = current
                    elif param == 'weight_decay':
                        training_agent.config['optimizer_kwargs'][
                            'weight_decay'] = current
                reset()
                training_agent.run_epochs(trial=0, epochs=epochs)
                cur_rank = compute_rank(training_agent.metrics)
                print(f"Rank was {cur_rank}")
                print('--')
                trust_buffer[index] = cur_rank
            # if np.less(trust_buffer, 0.).any():  # not done populating buffer
            #     continue
        # TODO only works for 2D cse
        # Will handle duplicates and take the last index
        min_index = tuple(np.argwhere(
            trust_buffer == np.min(trust_buffer))[-1])
        print(f'min_index: {min_index}')
        rank_history.append(np.min(trust_buffer))
        for axis, i in enumerate(min_index):
            mid = int(trust_buffer.shape[axis] / 2)
            trust_buffer = np.roll(trust_buffer, (i - mid) * -1, axis=axis)
        # TODO THIS ONLY WORKS FOR 2D MATRIX
        if min_index[0] == 0 or min_index[0] == trust_buffer.shape[0] - 1:
            trust_buffer[min_index[0], :] = -1.
        if min_index[1] == 0 or min_index[1] == trust_buffer.shape[1] - 1:
            trust_buffer[:, min_index[1]] = -1.
        print(f'trust_buffer: {trust_buffer}')
        scale_power = list(np.array(min_index) - 1)
        for i, param in enumerate(hyper_parameters.config.keys()):
            current = hyper_parameters.config[param].current * (
                hyper_parameters.config[param].scale ** scale_power[i])
            hyper_parameters.config[param].current = current
            # if np.isclose(cur_rank, 0.0):
            #     print("RANK ZERO BACK UP")
            #     hyper_parameters.config[param]['current'] *= \
            #         hyper_parameters.config[param]['scale'] ** 2
            #     continue
            # hyper_parameters.config[param]['buffer'].append(cur_rank)
        zeta = np.cumprod(rank_history) ** power
        if np.less(zeta[-1], min_delta):
            for param in hyper_parameters.config.keys():
                if np.isclose(hyper_parameters.config[param].scale, 1.,
                              atol=scale_delta):
                    hyper_parameters.config[param].stop = True
                hyper_parameters.config[param].count += 1
                hyper_parameters.config[param].scale = .5 * \
                    np.exp(-hyper_parameters.config[param].count * 2) + 1

            # hyper_parameters.config[param]['current'] *= \
            #     hyper_parameters.config[param]['scale']
            # if param == 'init_lr':
            #     training_agent.config['init_lr'] = \
            #         hyper_parameters.config[param]['current']
            # elif param == 'weight_decay':
            #     training_agent.config['optimizer_kwargs']['weight_decay'
            #                                               ] = \
            #         hyper_parameters.config[param]['current']
        # for param, conf in hyper_parameters.config.items():
        #     print(param)
        #     print(conf)
        if all([config.stop for _, config in
                hyper_parameters.config.items()]):
            break
    # with (training_agent.output_path / 'lrrt.csv').open('w+') as f:
    #     f.write('lr,rank,msg\n')
    #     for (lr, rank, msg) in output_history:
    #         f.write(f'{lr},{rank},{msg}\n')
    return training_agent.config
