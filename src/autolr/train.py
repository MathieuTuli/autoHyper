"""
MIT License

Copyright (c) 2020 Mathieu Tuli

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
from typing import Tuple, Dict, Any, List
from datetime import datetime
from pathlib import Path

# import logging
import warnings
import time

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import numpy as np
import torch
import yaml

from .optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR
from .optim import get_optimizer_scheduler
from .lr_range_test import auto_lr
from .early_stop import EarlyStop
from .models import get_network
from .utils import parse_config
from .profiler import Profiler
from .metrics import Metrics
from .models.vgg import VGG
from .optim.sls import SLS
from .optim.sps import SPS
from .data import get_data
from .AdaS import AdaS


def args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("AutoLR Train Args")
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
        default='.autolr-data', type=str,
        help="Set data directory path: Default = '.autolr-data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='.autolr-output', type=str,
        help="Set output directory path: Default = '.autolr-output'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='.autolr-checkpoint', type=str,
        help="Set checkpoint directory path: Default = '.autolr-checkpoint")
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
    sub_parser.add_argument(
        '--gpu', default=None, type=int,
        help='GPU id to use.')
    sub_parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        dest='mpd',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training')
    sub_parser.set_defaults(mpd=False)
    sub_parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:23456', type=str,
        help='url used to set up distributed training')
    sub_parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help='distributed backend')
    sub_parser.add_argument(
        '--world-size', default=-1, type=int,
        help='number of nodes for distributed training')
    sub_parser.add_argument(
        '--rank', default=-1, type=int,
        help='node rank for distributed training')


class TrainingAgent:
    config: Dict[str, Any] = None
    train_loader = None
    test_loader = None
    train_sampler = None
    num_classes: int = None
    network: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None
    scheduler = None
    loss = None
    output_filename: Path = None

    def __init__(
            self,
            config_path: Path,
            device: str,
            output_path: Path,
            data_path: Path,
            checkpoint_path: Path,
            start_epoch: int = 0,
            resume: bool = False,
            gpu: int = None,
            ngpus_per_node: int = 0,
            world_size: int = -1,
            rank: int = -1,
            dist: bool = False,
            mpd: bool = False,
            dist_url: str = None,
            dist_backend: str = None) -> None:

        self.gpu = gpu
        self.mpd = mpd
        self.dist = dist
        self.rank = rank
        self.best_acc = 0.
        self.device = device
        self.dist_url = dist_url
        self.world_size = world_size
        self.start_epoch = start_epoch
        self.dist_backend = dist_backend
        self.ngpus_per_node = ngpus_per_node

        self.data_path = data_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path

        self.load_config(config_path, data_path)
        print("AutoLR: Experiment Configuration")
        print("-"*45)
        for k, v in self.config.items():
            if isinstance(v, list) or isinstance(v, dict):
                print(f"    {k:<20} {v}")
            else:
                print(f"    {k:<20} {v:<20}")
        print("-"*45)

    def load_config(self, config_path: Path, data_path: Path) -> None:
        with config_path.open() as f:
            self.config = config = parse_config(yaml.load(f))
        if self.device == 'cpu':
            warnings.warn("Using CPU will be slow")
        elif self.dist:
            if self.gpu is not None:
                config['mini_batch_size'] = int(
                    config['min_batch_size'] / self.ngpus_per_node)
                config['num_workers'] = int(
                    (config['num_workers'] + self.ngpus_per_node - 1) /
                    self.ngpus_per_node)
        self.train_loader, self.train_sampler, \
            self.test_loader, self.num_classes = get_data(
                name=config['dataset'], root=data_path,
                mini_batch_size=config['mini_batch_size'],
                num_workers=config['num_workers'],
                dist=self.dist)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.gpu) if \
            config['loss'] == 'cross_entropy' else None
        if np.less(float(config['early_stop_threshold']), 0):
            print("AutoLR: Notice: early stop will not be used as it was " +
                  f"set to {config['early_stop_threshold']}, " +
                  "training till completion")
        elif config['optimizer'] != 'SGD' and \
                config['lr_scheduler'] != 'AdaS':
            print("AutoLR: Notice: early stop will not be used as it is not " +
                  "SGD with AdaS, training till completion")
            config['early_stop_threshold'] = -1.
        self.early_stop = EarlyStop(
            patience=int(config['early_stop_patience']),
            threshold=float(config['early_stop_threshold']))
        cudnn.benchmark = True
        # self.reset()

    def reset(self, learning_rate: float) -> None:
        self.performance_statistics = dict()
        self.network = get_network(name=self.config['network'],
                                   num_classes=self.num_classes)
        self.metrics = Metrics(list(self.network.parameters()),
                               p=self.config['p'])
        # TODO add other parallelisms
        if self.device == 'cpu':
            print("Resetting cpu-based network")
        elif self.dist:
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.network.cuda(self.gpu)
                self.network = torch.nn.parallel.DsitributedDataParallel(
                    self.network,
                    device_ids=[self.gpu])
            else:
                self.network.cuda()
                self.network = torch.nn.parallel.DsitributedDataParallel(
                    self.network)
        elif self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            self.network = self.network.cuda(self.gpu)
        else:
            if isinstance(self.network, VGG):
                self.network.features = torch.nn.DataParallel(
                    self.network.features)
                self.network.cuda()
            else:
                self.network = torch.nn.DataParallel(self.network)
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optim_method=self.config['optimizer'],
            lr_scheduler=self.config['scheduler'],
            init_lr=learning_rate,
            net_parameters=self.network.parameters(),
            listed_params=list(self.network.parameters()),
            train_loader_len=len(self.train_loader),
            max_epochs=self.config['max_epochs'],
            **self.config['kwargs'])
        self.early_stop.reset()

    def train(self) -> None:
        if not isinstance(self.config['init_lr'], list):
            list_lr = [self.config['init_lr']]
        else:
            list_lr = self.config['init_lr']
        for learning_rate in list_lr:
            if learning_rate == 'auto':
                learning_rate = auto_lr(
                    data_path=self.data_path, output_path=self.output_path,
                    device=self.device)
            lr_output_path = self.output_path / f'lr-{learning_rate}'
            lr_output_path.mkdir(exist_ok=True, parents=True)
            for trial in range(self.config['n_trials']):
                self.output_filename = \
                    "results_" +\
                    f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_" +\
                    f"trial=AdaS_trial={trial}_" +\
                    f"network={self.config['network']}_" +\
                    f"dataset={self.config['dataset']}" +\
                    f"optimizer={self.config['optimizer']}_" +\
                    f"scheduler={self.config['scheduler']}_" +\
                    f"learning_rate={learning_rate}_" +\
                    '_'.join([f"{k}={v}" for k, v in
                              self.config['kwargs'].items()]) +\
                    ".csv".replace(' ', '-')
                self.output_filename = str(
                    lr_output_path / self.output_filename)
                stats_filename = self.output_filename.replace(
                    'results', 'stats')
                Profiler.filename = lr_output_path / stats_filename
                self.reset(learning_rate)
                epochs = range(self.start_epoch, self.start_epoch +
                               self.config['max_epochs'])
                self.run_epochs(trial, epochs)
                Profiler.stream = None

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        for epoch in epochs:
            if self.dist:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss, (train_acc1, train_acc5) = \
                self.epoch_iteration(trial, epoch)
            test_loss, (test_acc1, test_acc5) = self.validate(epoch)
            end_time = time.time()
            if isinstance(self.scheduler, StepLR):
                self.scheduler.step()
            total_time = time.time()
            scheduler_string = f" w/ {self.config['scheduler']}" if \
                self.scheduler is not None else ''
            print(
                f"{self.config['optimizer']}{scheduler_string} " +
                f"on {self.config['dataset']}: " +
                f"T {trial + 1}/{self.config['n_trials']} | " +
                f"E {epoch + 1}/{epochs[-1] + 1} Ended | " +
                "E Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (total_time - start_time) * (epochs[-1] - epoch)),
                "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                    train_loss,
                    train_acc1 * 100) +
                "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(
                    test_loss,
                    test_acc1 * 100))
            df = pd.DataFrame(data=self.performance_statistics)

            df.to_csv(self.output_filename)
            if self.early_stop(train_loss):
                print("AutoLR: Early stop activated.")
                break

    # @Profiler
    def epoch_iteration(self, trial: int, epoch: int):
        # logging.info(f"Adas: Train: Epoch: {epoch}")
        # global net, performance_statistics, metrics, adas, config
        self.network.train()
        train_loss = 0
        top1 = AverageMeter()
        top5 = AverageMeter()
        # correct = 0
        # total = 0

        """train CNN architecture"""
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # start = time.time()
            # print(f'{batch_idx} / {len(train_loader)}')
            if self.gpu is not None:
                inputs = inputs.cuda(self.gpu, non_blocking=True)
            if self.device == 'cuda':
                targets = targets.cuda(self.gpu, non_blocking=True)
            # inputs, targets = inputs.to(self.device), targets.to(self.device)
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            self.optimizer.zero_grad()
            if isinstance(self.optimizer, SLS):
                def closure():
                    outputs = self.network(inputs)
                    loss = self.criterion(outputs, targets)
                    return loss, outputs
                loss, outputs = self.optimizer.step(closure=closure)
            else:
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if isinstance(self.scheduler, AdaS):
                    self.optimizer.step(self.metrics.layers_index_todo,
                                        self.scheduler.lr_vector)
                elif isinstance(self.optimizer, SPS):
                    self.optimizer.step(loss=loss)
                else:
                    self.optimizer.step()

            train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            top1.update(acc1.cpu().item())
            top5.update(acc5.cpu().item())
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
        self.performance_statistics[f'train_acc1_epoch_{epoch}'] = \
            top1.avg / 100.
        self.performance_statistics[f'train_acc5_epoch_{epoch}'] = \
            top5.avg / 100.
        self.performance_statistics[f'train_loss_epoch_{epoch}'] = \
            train_loss / (batch_idx + 1)

        io_metrics = self.metrics.evaluate(epoch)
        self.performance_statistics[f'in_S_epoch_{epoch}'] = \
            io_metrics.input_channel_S
        self.performance_statistics[f'out_S_epoch_{epoch}'] = \
            io_metrics.output_channel_S
        self.performance_statistics[f'fc_S_epoch_{epoch}'] = io_metrics.fc_S
        self.performance_statistics[f'in_rank_epoch_{epoch}'] = \
            io_metrics.input_channel_rank
        self.performance_statistics[f'out_rank_epoch_{epoch}'] = \
            io_metrics.output_channel_rank
        self.performance_statistics[f'fc_rank_epoch_{epoch}'] = \
            io_metrics.fc_rank
        self.performance_statistics[f'in_condition_epoch_{epoch}'] = \
            io_metrics.input_channel_condition

        self.performance_statistics[f'out_condition_epoch_{epoch}'] = \
            io_metrics.output_channel_condition
        # if GLOBALS.ADAS is not None:
        if isinstance(self.scheduler, AdaS):
            lrmetrics = self.scheduler.step(epoch, self.metrics)
            self.performance_statistics[f'rank_velocity_epoch_{epoch}'] = \
                lrmetrics.rank_velocity
            self.performance_statistics[f'learning_rate_epoch_{epoch}'] = \
                lrmetrics.r_conv
        else:
            # if GLOBALS.CONFIG['optim_method'] == 'SLS' or \
            #         GLOBALS.CONFIG['optim_method'] == 'SPS':
            if isinstance(self.optimizer, SLS) or isinstance(
                    self.optimizer, SPS):
                self.performance_statistics[
                    f'learning_rate_epoch_{epoch}'] = \
                    self.optimizer.state['step_size']
            else:
                self.performance_statistics[
                    f'learning_rate_epoch_{epoch}'] = \
                    self.optimizer.param_groups[0]['lr']
        return train_loss / (batch_idx + 1), (top1.avg / 100.,
                                              top5.avg / 100.)

    def validate(self, epoch: int):
        self.network.eval()
        test_loss = 0
        # correct = 0
        # total = 0
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                # inputs, targets = \
                #     inputs.to(self.device), targets.to(self.device)
                if self.gpu is not None:
                    inputs = inputs.cuda(self.gpu, non_blocking=True)
                if self.device == 'cuda':
                    targets = targets.cuda(self.gpu, non_blocking=True)
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                # _, predicted = outputs.max(1)
                # total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(acc1.cpu().item())
                top5.update(acc5.cpu().item())

        # Save checkpoint.
        # acc = 100. * correct / total
        # if acc > self.best_acc:
        #     # print('Adas: Saving checkpoint...')
        #     state = {
        #         'net': self.network.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch + 1,
        #     }
        #     if not isinstance(self.scheduler, AdaS):
        #         state['historical_io_metrics'] = \
        #             self.metrics.historical_metrics
        #     torch.save(state, str(self.checkpoint_path / 'ckpt.pth'))
        #     self.best_acc = acc
        self.performance_statistics[f'test_acc1_epoch_{epoch}'] = (
            top1.avg / 100.)
        self.performance_statistics[f'test_acc5_epoch_{epoch}'] = (
            top5.avg / 100.)
        self.performance_statistics[f'test_loss_epoch_{epoch}'] = \
            test_loss / (batch_idx + 1)
        return test_loss / (batch_idx + 1), (top1.avg / 100,
                                             top5.avg / 100)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_dirs(args: APNamespace) -> Tuple[Path, Path, Path, Path]:
    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    checkpoint_path = root_path / Path(args.checkpoint).expanduser()

    if not config_path.exists():
        raise ValueError(f"AutoLR: Config path {config_path} does not exist")
    if not data_path.exists():
        print(f"AutoLR: Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"AutoLR: Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not checkpoint_path.exists():
        if args.resume:
            raise ValueError("AutoLR: Cannot resume from checkpoint without " +
                             "specifying checkpoint dir")
        checkpoint_path.mkdir(exist_ok=True, parents=True)
    return config_path, output_path, data_path, checkpoint_path


def main(args: APNamespace):
    print("AutoLR: Argument Parser Options")
    print("-"*45)
    for arg in vars(args):
        attr = getattr(args, arg)
        attr = attr if attr is not None else "None"
        print(f"    {arg:<20}: {attr:<40}")
    print("-"*45)
    args.config_path, args.output_path, \
        args.data_path, args.checkpoint_path = setup_dirs(args)
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.mpd or args.world_size > 1
    if args.mpd:
        args.world_size *= args.ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu: int, ngpus_per_node: int, args: APNamespace):
    args.gpu = gpu
    if args.distributed:
        if args.mpd:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    training_agent = TrainingAgent(
        config_path=args.config_path,
        device=device,
        output_path=args.output_path,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        start_epoch=0,
        resume=args.resume,
        gpu=args.gpu,
        ngpus_per_node=ngpus_per_node,
        world_size=args.world_size,
        rank=args.rank,
        dist=args.distributed,
        mpd=args.mpd,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend)
    print(f"AutoLR: Pytorch device is set to {training_agent.device}")
    training_agent.train()
