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
from typing import Tuple
from pathlib import Path

# import logging
import time

import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch
import yaml

from .optim import get_optimizer_scheduler
from .early_stop import EarlyStop
from .utils import parse_config
from .profiler import Profiler
from .metrics import Metrics
from .models import get_net
from .test import test_main
from .data import get_data
from .optim.sls import SLS
from .optim.sps import SPS
from .AdaS import AdaS

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


def main(args: APNamespace):
    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    # global checkpoint_path, config
    GLOBALS.CHECKPOINT_PATH = root_path / Path(args.checkpoint).expanduser()

    if not config_path.exists():
        # logging.critical(f"AdaS: Config path {config_path} does not exist")
        print(f"AdaS: Config path {config_path} does not exist")
        raise ValueError
    if not data_path.exists():
        print(f"AdaS: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"AdaS: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not GLOBALS.CHECKPOINT_PATH.exists():
        if args.resume:
            print(f"AdaS: Cannot resume from checkpoint without specifying " +
                  "checkpoint dir")
            raise ValueError
        GLOBALS.CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)
    with config_path.open() as f:
        GLOBALS.CONFIG = parse_config(yaml.load(f))
    print("Adas: Argument Parser Options")
    print("-"*45)
    print(f"    {'config':<20}: {args.config:<40}")
    print(f"    {'data':<20}: {str(Path(args.root) / args.data):<40}")
    print(f"    {'output':<20}: {str(Path(args.root) / args.output):<40}")
    print(f"    {'checkpoint':<20}: " +
          f"{str(Path(args.root) / args.checkpoint):<40}")
    print(f"    {'root':<20}: {args.root:<40}")
    print(f"    {'resume':<20}: {'True' if args.resume else 'False':<20}")
    print(f"    {'cpu':<20}: {'True' if args.cpu else 'False':<20}")
    print("\nAdas: Train: Config")
    print(f"    {'Key':<20} {'Value':<20}")
    print("-"*45)
    for k, v in GLOBALS.CONFIG.items():
        print(f"    {k:<20} {v:<20}")
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"AdaS: Pytorch device is set to {device}")
    # global best_acc
    GLOBALS.BEST_ACC = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if np.less(float(GLOBALS.CONFIG['early_stop_threshold']), 0):
        print(
            "AdaS: Notice: early stop will not be used as it was set to " +
            f"{GLOBALS.CONFIG['early_stop_threshold']}, training till " +
            "completion")

    learning_rate = GLOBALS.CONFIG['init_lr']
    # if config['init_lr'] == 'auto':
    #     learning_rate = lr_range_test(
    #         root=output_path,
    #         config=config
    #         learning_rate=learning_rate)
    for trial in range(GLOBALS.CONFIG['n_trials']):
        if GLOBALS.CONFIG['lr_scheduler'] == 'AdaS':
            filename = \
                f"stats_{GLOBALS.CONFIG['optim_method']}_AdaS_trial={trial}" +\
                f"_beta={GLOBALS.CONFIG['beta']}_initlr={learning_rate}_" +\
                f"net={GLOBALS.CONFIG['network']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.csv"
        else:
            filename = \
                f"stats_{GLOBALS.CONFIG['optim_method']}_" +\
                f"{GLOBALS.CONFIG['lr_scheduler']}_" +\
                f"trial={trial}_initlr={learning_rate}" +\
                f"net={GLOBALS.CONFIG['network']}_dataset=" + \
                f"{GLOBALS.CONFIG['dataset']}.csv"
        Profiler.filename = output_path / filename
        # Data
        # logging.info("Adas: Preparing Data")
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
        # if GLOBALS.CONFIG['lr_scheduler'] == 'AdaS':
        #     GLOBALS.ADAS = AdaS(parameters=list(GLOBALS.NET.parameters()),
        #                         beta=GLOBALS.CONFIG['beta'],
        #                         zeta=GLOBALS.CONFIG['zeta'],
        #                         init_lr=learning_rate,
        #                         min_lr=float(GLOBALS.CONFIG['min_lr']),
        #                         p=GLOBALS.CONFIG['p'])

        GLOBALS.NET = GLOBALS.NET.to(device)

        # global criterion
        GLOBALS.CRITERION = get_loss(GLOBALS.CONFIG['loss'])

        optimizer, scheduler = get_optimizer_scheduler(
            net_parameters=GLOBALS.NET.parameters(),
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

        if args.resume:
            # Load checkpoint.
            print("Adas: Resuming from checkpoint...")
            checkpoint = torch.load(str(GLOBALS.CHECKPOINT_PATH / 'ckpt.pth'))
            # if checkpoint_path.is_dir():
            #     checkpoint = torch.load(str(checkpoint_path / 'ckpt.pth'))
            # else:
            #     checkpoint = torch.load(str(checkpoint_path))
            GLOBALS.NET.load_state_dict(checkpoint['net'])
            GLOBALS.BEST_ACC = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            # if GLOBALS.ADAS is not None:
            if isinstance(scheduler, AdaS):
                GLOBALS.METRICS.historical_metrics = \
                    checkpoint['historical_io_metrics']

        # model_parameters = filter(lambda p: p.requires_grad,
        #                           net.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print(params)
        epochs = range(start_epoch, start_epoch + GLOBALS.CONFIG['max_epoch'])
        run_epochs(trial, epochs, train_loader, test_loader,
                   device, optimizer, scheduler, output_path)

        # Needed to reset profiler file stream
        Profiler.stream = None
    return


def run_epochs(trial, epochs, train_loader, test_loader,
               device, optimizer, scheduler, output_path):
    for epoch in epochs:
        start_time = time.time()
        # print(f"AdaS: Epoch {epoch}/{epochs[-1]} Started.")
        train_loss, train_accuracy, test_loss, test_accuracy = \
            epoch_iteration(trial,
                            train_loader, test_loader,
                            epoch, device, optimizer, scheduler)
        end_time = time.time()
        if GLOBALS.CONFIG['lr_scheduler'] == 'StepLR':
            scheduler.step()
        total_time = time.time()
        print(
            f"AdaS: Trial {trial}/{GLOBALS.CONFIG['n_trials'] - 1} | " +
            f"Epoch {epoch}/{epochs[-1]} Ended | " +
            "Total Time: {:.3f}s | ".format(total_time - start_time) +
            "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
            "~Time Left: {:.3f}s | ".format(
                (total_time - start_time) * (epochs[-1] - epoch)),
            "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                train_loss,
                train_accuracy) +
            "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(test_loss,
                                                            test_accuracy))
        df = pd.DataFrame(data=GLOBALS.PERFORMANCE_STATISTICS)
        if GLOBALS.CONFIG['lr_scheduler'] == 'AdaS':
            xlsx_name = \
                f"{GLOBALS.CONFIG['optim_method']}_AdaS_trial={trial}_" +\
                f"beta={GLOBALS.CONFIG['beta']}_initlr=" +\
                f"{GLOBALS.CONFIG['init_lr']}_" +\
                f"net={GLOBALS.CONFIG['network']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"
        else:
            xlsx_name = \
                f"{GLOBALS.CONFIG['optim_method']}_" +\
                f"{GLOBALS.CONFIG['lr_scheduler']}_" +\
                f"trial={trial}_initlr={GLOBALS.CONFIG['init_lr']}" +\
                f"net={GLOBALS.CONFIG['network']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"

        df.to_excel(str(output_path / xlsx_name))
        if GLOBALS.EARLY_STOP(train_loss):
            print("AdaS: Early stop activated.")
            break


@Profiler
def epoch_iteration(trial, train_loader, test_loader, epoch: int,
                    device, optimizer, scheduler) -> Tuple[float, float]:
    # logging.info(f"Adas: Train: Epoch: {epoch}")
    # global net, performance_statistics, metrics, adas, config
    GLOBALS.NET.train()
    train_loss = 0
    correct = 0
    total = 0

    """train CNN architecture"""
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if GLOBALS.CONFIG['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + batch_idx / len(train_loader))
        optimizer.zero_grad()
        # if GLOBALS.CONFIG['optim_method'] == 'SLS':
        if isinstance(optimizer, SLS):
            def closure():
                outputs = GLOBALS.NET(inputs)
                loss = GLOBALS.CRITERION(outputs, targets)
                return loss, outputs
            loss, outputs = optimizer.step(closure=closure)
        else:
            outputs = GLOBALS.NET(inputs)
            loss = GLOBALS.CRITERION(outputs, targets)
            loss.backward()
            # if GLOBALS.ADAS is not None:
            #     optimizer.step(GLOBALS.METRICS.layers_index_todo,
            #                    GLOBALS.ADAS.lr_vector)
            if isinstance(scheduler, AdaS):
                optimizer.step(GLOBALS.METRICS.layers_index_todo,
                               scheduler.lr_vector)
            # elif GLOBALS.CONFIG['optim_method'] == 'SPS':
            elif isinstance(optimizer, SPS):
                optimizer.step(loss=loss)
            else:
                optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if GLOBALS.CONFIG['lr_scheduler'] == 'OneCycleLR':
            scheduler.step()

        # progress_bar(batch_idx, len(train_loader),
        #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1),
        #                  100. * correct / total, correct, total))
    GLOBALS.PERFORMANCE_STATISTICS[f'train_acc_epoch_{epoch}'] = \
        float(correct / total)
    GLOBALS.PERFORMANCE_STATISTICS[f'train_loss_epoch_{epoch}'] = \
        train_loss / (batch_idx + 1)

    io_metrics = GLOBALS.METRICS.evaluate(epoch)
    GLOBALS.PERFORMANCE_STATISTICS[f'in_S_epoch_{epoch}'] = \
        io_metrics.input_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'out_S_epoch_{epoch}'] = \
        io_metrics.output_channel_S
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_S_epoch_{epoch}'] = \
        io_metrics.fc_S
    GLOBALS.PERFORMANCE_STATISTICS[f'in_rank_epoch_{epoch}'] = \
        io_metrics.input_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'out_rank_epoch_{epoch}'] = \
        io_metrics.output_channel_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'fc_rank_epoch_{epoch}'] = \
        io_metrics.fc_rank
    GLOBALS.PERFORMANCE_STATISTICS[f'in_condition_epoch_{epoch}'] = \
        io_metrics.input_channel_condition

    GLOBALS.PERFORMANCE_STATISTICS[f'out_condition_epoch_{epoch}'] = \
        io_metrics.output_channel_condition
    # if GLOBALS.ADAS is not None:
    if isinstance(scheduler, AdaS):
        lrmetrics = scheduler.step(epoch, GLOBALS.METRICS)
        GLOBALS.PERFORMANCE_STATISTICS[f'rank_velocity_epoch_{epoch}'] = \
            lrmetrics.rank_velocity
        GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
            lrmetrics.r_conv
    else:
        # if GLOBALS.CONFIG['optim_method'] == 'SLS' or \
        #         GLOBALS.CONFIG['optim_method'] == 'SPS':
        if isinstance(optimizer, SLS) or isinstance(optimizer, SPS):
            GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.state['step_size']
        else:
            GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.param_groups[0]['lr']
    test_loss, test_accuracy = test_main(test_loader, epoch, device)
    return (train_loss / (batch_idx + 1), 100. * correct / total,
            test_loss, test_accuracy)


if __name__ == "__main__":
    ...
