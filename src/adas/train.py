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

# import logging

import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import torch
import yaml

from .train_support import run_epochs
from .optim import get_optimizer_scheduler
from .lr_range_test import auto_lr
from .early_stop import EarlyStop
from .utils import parse_config
from .profiler import Profiler
from .metrics import Metrics
from .models import get_net
from .data import get_data
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
        raise ValueError(f"AdaS: Config path {config_path} does not exist")
    if not data_path.exists():
        print(f"AdaS: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"AdaS: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not GLOBALS.CHECKPOINT_PATH.exists():
        if args.resume:
            raise ValueError(f"AdaS: Cannot resume from checkpoint without " +
                             "specifying checkpoint dir")
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
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    if GLOBALS.CONFIG['init_lr'] == 'auto':
        GLOBALS.CONFIG['init_lr'] = auto_lr(
            data_path=data_path, output_path=output_path, device=device)
    for k, v in GLOBALS.CONFIG.items():
        if isinstance(v, list):
            print(f"    {k:<20} {v}")
        else:
            print(f"    {k:<20} {v:<20}")
    print(f"AdaS: Pytorch device is set to {device}")
    # global best_acc
    GLOBALS.BEST_ACC = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if np.less(float(GLOBALS.CONFIG['early_stop_threshold']), 0):
        print(
            "AdaS: Notice: early stop will not be used as it was set to " +
            f"{GLOBALS.CONFIG['early_stop_threshold']}, training till " +
            "completion")
    elif GLOBALS.CONFIG['optim_method'] != 'SGD' and \
            GLOBALS.CONFIG['lr_scheduler'] != 'AdaS':
        print(
            "AdaS: Notice: early stop will not be used as it is not SGD with" +
            " AdaS, training till completion")
        GLOBALS.CONFIG['early_stop_threshold'] = -1.

    # if config['init_lr'] == 'auto':
    #     learning_rate = lr_range_test(
    #         root=output_path,
    #         config=config
    #         learning_rate=learning_rate)
    base_output_path = output_path
    if not isinstance(GLOBALS.CONFIG['init_lr'], list):
        list_lr = [GLOBALS.CONFIG['init_lr']]
    else:
        list_lr = GLOBALS.CONFIG['init_lr']
    for learning_rate in list_lr:
        GLOBALS.CONFIG['init_lr'] = learning_rate
        output_path = base_output_path / f'lr-{learning_rate}'
        output_path.mkdir(exist_ok=True, parents=True)
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
                mini_batch_size=GLOBALS.CONFIG['mini_batch_size'],
                num_workers=GLOBALS.CONFIG['num_workers'])
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

            if args.resume:
                # Load checkpoint.
                print("Adas: Resuming from checkpoint...")
                checkpoint = torch.load(
                    str(GLOBALS.CHECKPOINT_PATH / 'ckpt.pth'))
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
            epochs = range(start_epoch, start_epoch +
                           GLOBALS.CONFIG['max_epoch'])
            run_epochs(trial, epochs, train_loader, test_loader,
                       device, optimizer, scheduler, output_path)

            # Needed to reset profiler file stream
            Profiler.stream = None
    return


if __name__ == "__main__":
    ...
