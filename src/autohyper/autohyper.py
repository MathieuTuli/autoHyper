
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
from pathlib import Path
import itertools

# import logging

import numpy as np

from .metrics import Metrics
from .components import HyperParameters


def compute_rank(metrics: Metrics) -> float:
    per_S_zero = np.mean([np.mean(np.isclose(
        metric.input_channel_S + metric.output_channel_S,
        0).astype(np.float16)) for
        metric in metrics.historical_metrics])
    return per_S_zero


def optimize(epoch_trainer: callable,
             hyper_parameters: HyperParameters,
             min_delta: float = 5e-3,
             scale_delta: float = 5e-3,
             epochs: Union[range, List[int]] = range(0, 5),
             power: float = 0.8,
             output_path: str = 'autohyper-output'):
    """
    @arguments:
        epoch_trainer: callable
            required arguments (
                hyper_parameters: Dict[str, float],
                epochs: iter(int),
            )
            must also return Metrics, computed using metrics.py:Metrics

        hyper_parameters: HyperParameters
            starting hyper-parameter config object. See components.py
                for default
        min_delta: float
            delta between successive ranks that defines plateauing
        scale_delta:
            delta absolute tolerance
        epochs: iter[int]
            integer iterable for number of epochs per trial. (fixed)
                to range(5)
        power: float
            regularization power for cummulative product
        output_path: str
            string path  of output directory for logging and results
    """
    cur_rank = -1
    establish_start = True
    auto_lr_path = Path(output_path)
    auto_lr_path.mkdir(exist_ok=True)
    # date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # for param in hyper_parameters.config:
    #     delta[param] = .1
    #     if param == 'init_lr':
    #         training_agent.config['init_lr'] = \
    #             hyper_parameters.config[param].current
    #     elif param == 'weight_decay':
    #         training_agent.config['optimizer_kwargs']['weight_decay'] = \
    #             hyper_parameters.config[param].current

    rank_history = list()
    num_hp = len(hyper_parameters)
    # defines the trust region
    # we will [scale-down, not-scale, scale-up]
    scale_powers = [-1, 0, 1]
    trust_region = list(itertools.product(
        *[scale_powers for i in range(num_hp)]))
    trust_buffer = np.full([len(scale_powers) for _ in range(num_hp)],
                           fill_value=-1., dtype=float)

    def reset():
        pass

    cur_train_params = {p: v.current for p, v in hyper_parameters.items()}
    while True:
        for scale_power in trust_region:
            index = tuple(np.array(scale_power) + 1)
            if np.less(trust_buffer[index], 0.):
                for i, param in enumerate(hyper_parameters):
                    if scale_power[i] == 1 and np.equal(
                            hyper_parameters[param].current, 0.):
                        current = 1e-7
                    else:
                        current = hyper_parameters[param].current * \
                            (hyper_parameters[param].scale **
                             scale_power[i])
                    cur_train_params[param] = current
                metrics = epoch_trainer(hyper_parameters=cur_train_params,
                                        epochs=epochs,)
                cur_rank = compute_rank(training_agent.metrics)
                trust_buffer[index] = cur_rank
        # TODO only works for 2D cse
        # Will handle duplicates and take the last index
        if establish_start and (trust_buffer < .85).all() and all(
            np.greater_equal(hp.current, hp.minimum)
                for k, hp in hyper_parameters.items()):
            index = tuple(np.argwhere(
                trust_buffer == np.max(trust_buffer))[0])
            if index == (1, 1):
                index = (0, 0)
        else:
            establish_start = False
            index = tuple(np.argwhere(
                trust_buffer == np.min(trust_buffer))[-1])
            rank_history.append(np.min(trust_buffer))
            if index == (1, 1):
                index = (2, 2)
        for axis, i in enumerate(index):
            mid = int(trust_buffer.shape[axis] / 2)
            trust_buffer = np.roll(trust_buffer, (i - mid) * -1, axis=axis)
        # TODO THIS ONLY WORKS FOR 2D MATRIX
        if index[0] == 0 or index[0] == trust_buffer.shape[0] - 1:
            trust_buffer[index[0], :] = -1.
        if index[1] == 0 or index[1] == trust_buffer.shape[1] - 1:
            trust_buffer[:, index[1]] = -1.
        scale_power = list(np.array(index) - 1)
        for i, param in enumerate(hyper_parameters):
            if scale_power[i] == 1 and np.equal(
                    hyper_parameters[param].current, 0.):
                current = 1e-7
            else:
                current = hyper_parameters[param].current * (
                    hyper_parameters[param].scale ** scale_power[i])
            hyper_parameters[param].current = current
        if establish_start:
            continue
        zeta = np.cumprod(rank_history) ** power
        if np.less(zeta[-1], min_delta):
            for param in hyper_parameters:
                if np.isclose(hyper_parameters[param].scale, 1.,
                              atol=scale_delta):
                    hyper_parameters[param].stop = True
                hyper_parameters[param].scale = \
                    (hyper_parameters[param].scale - 1.) * \
                    np.exp(-2.5) + 1

        if all([param.stop for param in
                hyper_parameters.values()]):
            break
    return hyper_parameters
