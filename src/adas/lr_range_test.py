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
from argparse import Namespace as APNamespace, _SubParsersAction
from pathlib import Path
from typing import List

# import logging

import torch.backends.cudnn as cudnn
import numpy as np
import torch
import yaml

from .optim import get_optimizer_scheduler
from .train_support import run_epochs
from .early_stop import EarlyStop
from .utils import parse_config
from .profiler import Profiler
from .metrics import Metrics
from .models import get_net
from .data import get_data

from . import global_vars as GLOBALS


def args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("AdaS Train Args")
    # print("---------------------------------\n")
    # sub_parser.add_argument(
    #     '-vv', '--very-verbose', action='store_true',
    #     dest='very_verbose',
    #     help="Set flask debug mode")
    # sub_parser.add_argument(
    #     '-v', '--verbose', action='store_true',
    #     dest='verbose',
    #     help="Set flask debug mode")
    # sub_parser.set_defaults(verbose=False)
    # sub_parser.set_defaults(very_verbose=False)
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    sub_parser.add_argument(
        '--data', dest='data',
        default='.adas-data', type=str,
        help="Set data directory path: Default = '.adas-data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='.adas-output', type=str,
        help="Set output directory path: Default = '.adas-output'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='.adas-checkpoint', type=str,
        help="Set checkpoint directory path: Default = '.adas-checkpoint")
    sub_parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.add_argument(
        '-r', '--resume', action='store_true',
        dest='resume',
        help="Flag: resume training from checkpoint")
    sub_parser.set_defaults(resume=False)
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training")
    sub_parser.set_defaults(cpu=False)


def get_loss(loss: str) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss() if loss == 'cross_entropy' else \
        None


def compute_rank() -> float:
    per_S_zero = np.mean([np.mean(np.isclose(
        metric.input_channel_S + metric.output_channel_S,
        0).astype(np.float16)) for
        metric in GLOBALS.METRICS.historical_metrics])
    return per_S_zero


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


def auto_lr(data_path: Path, output_path: Path, device: str):
    # ###### LR RANGE STUFF #######
    min_lr = 1e-4
    max_lr = 0.1
    num_split = 20
    learning_rates = np.geomspace(min_lr, max_lr, num_split)
    rank_history = list()
    lr_idx = 0
    min_delta = 5e-2
    rank_thresh = 0.93 * 0
    exit_counter = 0
    lr_delta = 3e-5
    output_history = list()
    first_run = True
    epochs = range(0, 5)
    set_threshold = False
    set_threshold_list = []
    historical_rate_of_change = list()
    ##############################
    # while lr_idx < len(learning_rates):
    cur_rank = 0.0
    auto_lr_path = output_path / 'auto-lr'
    auto_lr_path.mkdir(exist_ok=True)
    while True:
        if lr_idx == len(learning_rates) and not set_threshold:
            min_lr = learning_rates[-2]
            max_lr = float(learning_rates[-1]) * 1.5
            print("LR Range Test: Reached End")
            learning_rates = np.geomspace(min_lr, max_lr, num_split)
            rank_history = list()
            output_history.append(
                (learning_rates[lr_idx - 1], -1, 'end-reached'))
            historical_rate_of_change.append(
                (learning_rates[lr_idx - 1], -1, 'end-reached'))
            lr_idx = 0
            continue
        if np.less(np.abs(np.subtract(min_lr, max_lr)), lr_delta) and not set_threshold:
            print(
                f"LR Range Test Complete: LR Delta: Final LR Range is {min_lr}-{max_lr}")
            output_history.append(
                (learning_rates[lr_idx], cur_rank, 'exit-delta'))
            historical_rate_of_change.append(
                (learning_rates[lr_idx], cur_rank, 'exit-delta'))
            break
        # Data
        # logging.info("Adas: Preparing Data")
        GLOBALS.CONFIG['init_lr'] = learning_rates[lr_idx]
        print(f"Using LR: {GLOBALS.CONFIG['init_lr']}")
        train_loader, test_loader = get_data(
            root=data_path,
            dataset=GLOBALS.CONFIG['dataset'],
            mini_batch_size=GLOBALS.CONFIG['mini_batch_size'])
        # global performance_statistics, net, metrics, adas
        GLOBALS.PERFORMANCE_STATISTICS = {}

        # logging.info("AdaS: Building Model")
        GLOBALS.NET = get_net(
            GLOBALS.CONFIG['network'], num_classes=10 if
            GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
            GLOBALS.CONFIG['dataset'] == 'CIFAR100'
            else 1000 if GLOBALS.CONFIG['dataset'] == 'ImageNet' else 10)
        GLOBALS.METRICS = Metrics(list(GLOBALS.NET.parameters()),
                                  p=GLOBALS.CONFIG['p'])

        GLOBALS.NET = GLOBALS.NET.to(device)

        # global criterion
        GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])

        optimizer, scheduler = get_optimizer_scheduler(
            net_parameters=GLOBALS.NET.parameters(),
            listed_params=list(GLOBALS.NET.parameters()),
            # init_lr=learning_rate,
            # optim_method=GLOBALS.CONFIG['optim_method'],
            # lr_scheduler=GLOBALS.CONFIG['lr_scheduler'],
            train_loader_len=len(train_loader),
            config=GLOBALS.CONFIG)
        # max_epochs=int(GLOBALS.CONFIG['max_epoch']))
        GLOBALS.EARLY_STOP = EarlyStop(
            patience=int(GLOBALS.CONFIG['early_stop_patience']),
            threshold=float(GLOBALS.CONFIG['early_stop_threshold']))

        if device == 'cuda':
            GLOBALS.NET = torch.nn.DataParallel(GLOBALS.NET)
            cudnn.benchmark = True

        run_epochs(0, epochs, train_loader, test_loader,
                   device, optimizer, scheduler, auto_lr_path)

        Profiler.stream = None

        # ###### LR RANGE STUFF #######
        if set_threshold:
            cur_rank = compute_rank()
            print(f"LR Range Test: Cur lr: {learning_rates[lr_idx]}")
            print(f"LR Range Test: Cur %: {cur_rank}")
            print("LR Range Test: Setting threshold still...")
            lr_idx += 1
            if np.isclose(cur_rank, 1.0):
                set_threshold_list = np.array(set_threshold_list)
                # non_zero_thresh =\
                #     (np.max(set_threshold_list) +
                #      np.min(set_threshold_list[
                #          np.greater(set_threshold_list, 0.9)])) / 2.
                rank_thresh = np.min(set_threshold_list[
                    np.greater(set_threshold_list, 0.9)])
                print(f"LR Range Test: Threshold set to {rank_thresh}")
                set_threshold = False
                lr_idx = 0
                continue
            set_threshold_list.append(cur_rank)
            continue
        cur_rank = compute_rank()
        rank_history.append(cur_rank)
        print(f"LR Range Test: Cur Space: {learning_rates}")
        print(f"LR Range Test: Cur lr: {learning_rates[lr_idx]}")
        print(f"LR Range Test: Cur %: {cur_rank}")
        # TODO minimum on min_lr?
        if np.isclose(cur_rank, 1.):
            output_history.append((learning_rates[lr_idx],
                                   cur_rank, 'all-zero'))
            historical_rate_of_change.append((learning_rates[lr_idx],
                                              cur_rank, 'all-zero'))
            min_lr *= 5
            if np.greater(min_lr, max_lr):
                max_lr = min_lr*1.5
            learning_rates = np.geomspace(min_lr, max_lr, num_split)
            rank_history = list()
            print("LR Range Test: All zero, new range: " +
                  f"{learning_rates}")
            lr_idx = 0
            continue
        if lr_idx == 0:
            if first_run and np.less(cur_rank, 0.5):
                output_history.append((learning_rates[lr_idx],
                                       cur_rank, 'early-cross'))
                historical_rate_of_change.append((learning_rates[lr_idx],
                                                  cur_rank, 'early-cross'))
                min_lr /= 10
                learning_rates = np.geomspace(min_lr, max_lr, num_split)
                rank_history = list()
                print("LR Range Test: Crossed thresh early, new range: " +
                      f"{learning_rates}")
                # lr_idx = 0 ; redundant
                continue
        else:
            first_run = False
            if np.isclose(cur_rank, 0.0):
                output_history.append((learning_rates[lr_idx],
                                       cur_rank, 'reach-100'))
                historical_rate_of_change.append((learning_rates[lr_idx],
                                                  cur_rank, 'reach-100'))
                min_lr = learning_rates[max(lr_idx - 2, 0)]
                max_lr = learning_rates[lr_idx - 1] if \
                    learning_rates[lr_idx - 1] != min_lr else \
                    (min_lr + learning_rates[lr_idx]) / 2.
                learning_rates = np.geomspace(min_lr, max_lr, num_split)
                rank_history = list()
                print("LR Range Test: Reached 100% non zero, new range: " +
                      f"{learning_rates}")
                lr_idx = 0
                continue
            # if np.greater(cur_rank, rank_thresh):
            rate_of_change = compute_rate_of_change(rank_history)
            historical_rate_of_change.append(
                (learning_rates[lr_idx], cur_rank, rate_of_change[-1]))

            # delta = np.subtract(
            #     cur_rank, rank_history[lr_idx - 1])
            # if np.less(rate_of_change[-1], min_delta):
            if len(rank_history) > 3 and \
                    np.isclose(rank_history[-1], rank_history[-2],
                               atol=min_delta) and \
                    np.isclose(rank_history[-2], rank_history[-3],
                               atol=min_delta) and \
                    np.less(cur_rank, 0.2):
                # if np.less(np.abs(delta), min_delta):
                output_history.append((learning_rates[lr_idx],
                                       cur_rank, 'plateau'))
                historical_rate_of_change.append((learning_rates[lr_idx],
                                                  cur_rank, 'plateau'))
                exit_counter += 1
                min_lr = learning_rates[max(lr_idx - 2, 0)]
                max_lr = learning_rates[lr_idx]
                learning_rates = np.geomspace(min_lr, max_lr, num_split)
                rank_history = list()
                print("LR Range Test: Hit Plateau, new range: " +
                      f"{learning_rates}")
                lr_idx = 0
                if exit_counter > 5:
                    output_history.append(
                        (learning_rates[lr_idx],
                            cur_rank, 'exit-counter'))
                    historical_rate_of_change.append(
                        (learning_rates[lr_idx],
                            cur_rank, 'exit-counter'))
                    break
                continue
        output_history.append((learning_rates[lr_idx],
                               cur_rank, 'nothing'))
        if exit_counter > 5:
            print(
                "LR Range Test Complete: Exit Counter: Final LR Range is " +
                f"{min_lr}-{max_lr}")
            output_history.append(
                (learning_rates[lr_idx], cur_rank, 'exit-counter'))
            break
        lr_idx += 1
    with (output_path / 'lrrt.csv').open('w+') as f:
        f.write('lr,rank,msg\n')
        for (lr, rank, msg) in output_history:
            f.write(f'{lr},{rank},{msg}\n')
    with (output_path / 'roc.csv').open('w+') as f:
        f.write('lr,rank,roc/msg\n')
        for (lr, rank, msg) in historical_rate_of_change:
            f.write(f'{lr},{rank},{msg}\n')
    return learning_rates[-1]


if __name__ == "__main__":
    ...
