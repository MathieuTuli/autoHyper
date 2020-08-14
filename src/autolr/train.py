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
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

# import logging

import torch.backends.cudnn as cudnn
import numpy as np
import torch
import yaml

from .lr_range_test import auto_lr, reset_experiment
from .optim import get_optimizer_scheduler
from .train_support import run_epochs
from .early_stop import EarlyStop
from .models import get_network
from .utils import parse_config
from .profiler import Profiler
from .data import get_data
from .AdaS import AdaS

from . import global_vars as GLOBALS


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


class TrainingAgent:
    config: Dict[str, Any] = None
    train_loader = None
    test_loader = None
    num_classes: int = None
    network: torch.nn.module = None
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
            resume: bool = False,) -> None:
        self.load_config(config_path, data_path)
        for k, v in self.config.items():
            if isinstance(v, list):
                print(f"    {k:<20} {v}")
            else:
                print(f"    {k:<20} {v:<20}")
        self.best_acc = 0.
        self.device = device
        self.start_epoch = start_epoch

        self.data_path = data_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path

    def load_config(self, config_path: Path, data_path: Path) -> None:
        with config_path.open() as f:
            self.config = config = parse_config(yaml.load(f))
        self.train_loader, self.test_loader, self.num_classes = get_data(
            name=config['dataset'], root=data_path,
            mini_batch_size=config['mini_batch_size'],
            num_workers=config['num_workers'])
        self.network = get_network(name=config['network'],
                                   num_classes=self.num_classes)
        self.network.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss() if \
            config['loss'] == 'cross_entropy' else None
        self.criterion.to(self.device)
        # self.optimizer, self.scheduler = get_optimizer_scheduler(
        #     optim_method=config['optimizer'],
        #     lr_scheduler=config['scheduler'],
        #     init_lr=config['init_lr'],
        #     net_parameters=self.network.parameters(),
        #     train_loader_len=len(self.train_loader),
        #     max_epochs=config['max_epochs'],
        #     **config['kwargs'])
        if np.less(float(config['early_stop_threshold']), 0):
            print("AutoLR: Notice: early stop will not be used as it was " +
                  f"set to {GLOBALS.CONFIG['early_stop_threshold']}, " +
                  "training till completion")
        elif config['optim_method'] != 'SGD' and \
                config['lr_scheduler'] != 'AdaS':
            print("AutoLR: Notice: early stop will not be used as it is not " +
                  "SGD with AdaS, training till completion")
            config['early_stop_threshold'] = -1.
        self.early_stop = EarlyStop(
            patience=int(config['early_stop_patience']),
            threshold=float(config['early_stop_threshold']))
        self.metrics = Metrics(list(self.network.parameters()), p=config['p'])
        self.performance_statistics = dict()

    def reset(self) -> None:
        if device == 'cuda':
            GLOBALS.NET = torch.nn.DataParallel(GLOBALS.NET)
            cudnn.benchmark = True
        if args.resume:
            # Load checkpoint.
            print("Adas: Resuming from checkpoint...")
            checkpoint = torch.load(
                str(GLOBALS.CHECKPOINT_PATH / 'ckpt.pth'))
            GLOBALS.NET.load_state_dict(checkpoint['net'])
            GLOBALS.BEST_ACC = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            # if GLOBALS.ADAS is not None:
            if isinstance(scheduler, AdaS):
                GLOBALS.METRICS.historical_metrics = checkpoint['historical_io_metrics']

    def train(self) -> None:
        if not isinstance(self.config['init_lr'], list):
            list_lr = [self.config['init_lr']]
        else:
            list_lr = GLOBALS.CONFIG['init_lr']
        for learning_rate in list_lr:
            # if learning_rate == 'auto':
            #     learning_rate = auto_lr(
            #         data_path=self.data_path, output_path=self.output_path,
            #         device=device)
            lr_output_path = self.output_path / f'lr-{learning_rate}'
            lr_output_path.mkdir(exist_ok=True, parents=True)
            for trial in range(GLOBALS.CONFIG['n_trials']):
                self.output_filename = \
                    f"results_date={datetime.now()}_" +\
                    f"trial=AdaS_trial={trial}_" +\
                    f"network={self.config['network']}_" +\
                    f"dataset={self.config['dataset']}" +\
                    f"optimizer={self.config['optimizer']}_" +\
                    f"scheduler={self.config['scheduler']}_" +\
                    f"learning_rate={learning_rate}_" +\
                    '_'.join([f"{k}={v} for k, v in config['kwargs'].items()])"
                    ".xlsx"
                stats_filename=f"stats_date={datetime.now()}_" +
                    f"trial=AdaS_trial={trial}_" +
                    f"network={self.config['network']}_" +
                    f"dataset={self.config['dataset']}" +
                    f"optimizer={self.config['optimizer']}_" +
                    f"scheduler={self.config['scheduler']}_" +
                    f"learning_rate={learning_rate}_" +
                    '_'.join([f"{k}={v} for k, v in config['kwargs'].items()])"
                    ".csv"
                Profiler.filename=lr_output_path / stats_filename
                self.reset()
                epochs=range(self.start_epoch, self.start_epoch +
                               self.config['max_epoch'])
                run_epochs(trial, epochs)
                Profiler.stream=None

    def run_epoch(self, trial: int, epochs: List[int]) -> None:
        for epoch in epochs:
            start_time = time.time()
            train_loss, train_accuracy, test_loss, test_accuracy=epoch_iteration(trial,
                                train_loader, test_loader,
                                epoch, device, optimizer, scheduler)
            end_time=time.time()
            if GLOBALS.CONFIG['lr_scheduler'] == 'StepLR':
                scheduler.step()
            total_time=time.time()
            print(
                f"AutoLR: Trial {trial}/{GLOBALS.CONFIG['n_trials'] - 1} | " +
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
            df=pd.DataFrame(data=GLOBALS.PERFORMANCE_STATISTICS)

            df.to_excel(xlsx_path)
            if GLOBALS.EARLY_STOP(train_loss):
                print("AutoLR: Early stop activated.")
                break


    @ Profiler
    def epoch_iteration(trial, train_loader, test_loader, epoch: int,
                        device, optimizer, scheduler) -> Tuple[float, float]:
        # logging.info(f"Adas: Train: Epoch: {epoch}")
        # global net, performance_statistics, metrics, adas, config
        GLOBALS.NET.train()
        train_loss=0
        correct=0
        total=0

        """train CNN architecture"""
        for batch_idx, (inputs, targets) in enumerate(train_loader):  # enumerate(tqdm(train_loader)):
            # start = time.time()
            # print(f'{batch_idx} / {len(train_loader)}')
            inputs, targets=inputs.to(device), targets.to(device)
            if GLOBALS.CONFIG['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
                scheduler.step(epoch + batch_idx / len(train_loader))
            optimizer.zero_grad()
            # if GLOBALS.CONFIG['optim_method'] == 'SLS':
            if isinstance(optimizer, SLS):
                def closure():
                    outputs=GLOBALS.NET(inputs)
                    loss=GLOBALS.CRITERION(outputs, targets)
                    return loss, outputs
                loss, outputs=optimizer.step(closure=closure)
            else:
                outputs=GLOBALS.NET(inputs)
                loss=GLOBALS.CRITERION(outputs, targets)
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
            _, predicted=outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if GLOBALS.CONFIG['lr_scheduler'] == 'OneCycleLR':
                scheduler.step()
            # progress_bar(batch_idx, len(train_loader),
            #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (batch_idx + 1),
            #                  100. * correct / total, correct, total))
        GLOBALS.PERFORMANCE_STATISTICS[f'train_acc_epoch_{epoch}']=float(
            correct / total)
        GLOBALS.PERFORMANCE_STATISTICS[f'train_loss_epoch_{epoch}']=train_loss / (
            batch_idx + 1)

        io_metrics=GLOBALS.METRICS.evaluate(epoch)
        GLOBALS.PERFORMANCE_STATISTICS[f'in_S_epoch_{epoch}']=io_metrics.input_channel_S
        GLOBALS.PERFORMANCE_STATISTICS[f'out_S_epoch_{epoch}']=io_metrics.output_channel_S
        GLOBALS.PERFORMANCE_STATISTICS[f'fc_S_epoch_{epoch}']=io_metrics.fc_S
        GLOBALS.PERFORMANCE_STATISTICS[f'in_rank_epoch_{epoch}']=io_metrics.input_channel_rank
        GLOBALS.PERFORMANCE_STATISTICS[f'out_rank_epoch_{epoch}']=io_metrics.output_channel_rank
        GLOBALS.PERFORMANCE_STATISTICS[f'fc_rank_epoch_{epoch}']=io_metrics.fc_rank
        GLOBALS.PERFORMANCE_STATISTICS[f'in_condition_epoch_{epoch}']=io_metrics.input_channel_condition

        GLOBALS.PERFORMANCE_STATISTICS[f'out_condition_epoch_{epoch}']=io_metrics.output_channel_condition
        # if GLOBALS.ADAS is not None:
        if isinstance(scheduler, AdaS):
            lrmetrics=scheduler.step(epoch, GLOBALS.METRICS)
            GLOBALS.PERFORMANCE_STATISTICS[f'rank_velocity_epoch_{epoch}']=lrmetrics.rank_velocity
            GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}']=lrmetrics.r_conv
        else:
            # if GLOBALS.CONFIG['optim_method'] == 'SLS' or \
            #         GLOBALS.CONFIG['optim_method'] == 'SPS':
            if isinstance(optimizer, SLS) or isinstance(optimizer, SPS):
                GLOBALS.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}']=optimizer.state['step_size']
            else:
                GLOBALS.PERFORMANCE_STATISTICS[
                    f'learning_rate_epoch_{epoch}']=optimizer.param_groups[0]['lr']
        test_loss, test_accuracy=test_main(test_loader, epoch, device)
        return (train_loss / (batch_idx + 1), 100. * correct / total,
                test_loss, test_accuracy)


def setup_dirs(args: APNamespace) -> Tuple[Path, Path, Path, Path]:
    root_path=Path(args.root).expanduser()
    config_path=Path(args.config).expanduser()
    data_path=root_path / Path(args.data).expanduser()
    output_path=root_path / Path(args.output).expanduser()
    checkpoint_path=root_path / Path(args.checkpoint).expanduser()

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
    print("Adas: Argument Parser Options")
    print("-"*45)
    for arg in vars(args):
        print(f"    {arg:,20}: {getattr(args, arg)<40}")
    print("-"*45)
    config_path, output_path, data_path, checkpoint_path=setup_dirs(args)
    training_agent=TrainingAgent(
        config_path=config_path,
        device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu',
        output_path=output_path,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        start_epoch=0,
        resume=args.resume)
    for k, v in GLOBALS.CONFIG.items():
        if isinstance(v, list):
            print(f"    {k:<20} {v}")
        else:
            print(f"    {k:<20} {v:<20}")
    print(f"AutoLR: Pytorch device is set to {training_agent.device}")

    training_agent.train()



if __name__ == "__main__":
    ...
