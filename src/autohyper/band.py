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
from typing import Union, List, Dict, Any
from datetime import datetime
import pickle
import sys

# import logging

from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS
import numpy as np
import torch

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


class PyTorchWorker(Worker):
    def __init__(self,  training_agent, num_trials, **kwargs):
        super().__init__(**kwargs)
        self.training_agent = training_agent
        self.num_trials = num_trials
        self.count = 0

    def compute(self, config, budget, *args, **kwargs):
        if self.count == self.num_trials:
            return ({'loss': 10000,  # remember: HpBandSter always minimizes!
                     'info': {
                         'test accuracy': 10000,
                         'validation accuracy': 10000, }
                     })
        self.count += 1
        self.training_agent.config['init_lr'] = config['lr']
        self.training_agent.config['optimizer_kwargs']['weight_decay'] = \
            config['weight_decay']
        self.training_agent.reset(self.training_agent.config)
        self.training_agent.output_filename = self.training_agent.output_path / 'auto-lr'
        self.training_agent.output_filename.mkdir(exist_ok=True, parents=True)
        date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        string_name = \
            "auto_lr_results_" +\
            f"date={date}_" +\
            f"network={self.training_agent.config['network']}_" +\
            f"dataset={self.training_agent.config['dataset']}_" +\
            f"optimizer={self.training_agent.config['optimizer']}_" +\
            '_'.join([f"{k}={v}" for k, v in
                      self.training_agent.config['optimizer_kwargs'].items()]) +\
            f"scheduler={self.training_agent.config['scheduler']}_" +\
            '_'.join([f"{k}={v}" for k, v in
                      self.training_agent.config['scheduler_kwargs'].items()]) +\
            f"learning_rate={config['lr']}" +\
            ".csv".replace(' ', '-')
        self.training_agent.output_filename = str(
            self.training_agent.output_filename / string_name)
        self.training_agent.run_epochs(trial=0, epochs=range(5))

        test_loss, (test_acc1, test_acc5) = self.training_agent.validate(4)
        return ({
                'loss': test_loss,  # remember: HpBandSter always minimizes!
                'info': {
                    'test accuracy': test_acc1,
                    # 'train accuracy': train_acc1,
                    'validation accuracy': test_acc1,
                    # 'number of parameters': self.training_agent.network.number_of_parameters(),
                }

                })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter(
            'lr', lower=1e-8, upper=1e-1, default_value=0.1, log=True)

        wd = CSH.UniformFloatHyperparameter(
            'weight_decay', lower=0.0, upper=0.1, default_value=0.0, log=False)
        cs.add_hyperparameters([lr, wd])

        return cs


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
    # learning_rates = np.geomspace(min_lr, max_lr, num_split)
    auto_lr_path = training_agent.output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
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
                'AdaM': 30,  # 10 for 1d
                'AdaBound': 48,
                'AdaGrad': 65,  # 20 for 1d
                'SGD': 79},
            'CIFAR100': {
                'AdaM': 30,  # 10 for 1d
                'AdaBound': 48,
                'AdaGrad': 65,  # 20 for 1d
                'SGD': 79},
            'TinyImageNet': {
                'AdaM': 52,
                'AdaBound': 52,
                'AdaGrad': 62,
                'SGD': 81}

        }
    }
    num_trials = \
        trials[training_agent.config['network']
               ][training_agent.config['dataset']
                 ][training_agent.config['optimizer']]

    host = hpns.nic_name_to_host('lo')
    NS = hpns.NameServer(run_id=0, host=host, port=0,
                         working_directory=str(training_agent.output_path / 'auto-lr'))
    ns_host, ns_port = NS.start()

    worker = PyTorchWorker(training_agent, num_trials, run_id=0, host=host,
                           nameserver=ns_host, nameserver_port=ns_port,
                           timeout=120)
    worker.run(background=True)
    hyperband = HyperBand(configspace=worker.get_configspace(),
                          run_id=0,
                          host=host,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          # min_budget=1, max_budget=10
                          )
    res = hyperband.run(n_iterations=int(num_trials/7))
    config = res.get_id2config_mapping()
    best = res.get_incumbent_id()
    best_hps = config[best]['config']
    print(best_hps)

    hyperband.shutdown(shutdown_workers=True)
    NS.shutdown()

    training_agent.config['init_lr'] = best_hps['lr']
    training_agent.config['optimizer_kwargs']['weight_decay'] = best_hps['weight_decay']
    return training_agent.config
