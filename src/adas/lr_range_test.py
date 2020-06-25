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

# from .test import main as test_main
# from .utils import progress_bar
from adas.optim import get_optimizer_scheduler
from adas.early_stop import EarlyStop
from adas.profiler import Profiler
from adas.metrics import Metrics
from adas.models import get_net
from adas.data import get_data
from adas.AdaS import AdaS


net = None
performance_statistics = None
criterion = None
metrics = None
adas = None
early_stop = None
config = None
counter = 0


def args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("AdaS LR Range Test Args")
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
    # sub_parser.add_argument(
    #     '--beta', dest='beta',
    #     default=0.8, type=float,
    #     help="set beta hyper-parameter")
    # sub_parser.add_argument(
    #     '--zeta', dest='zeta',
    #     default=1.0, type=float,
    #     help="set zeta hyper-parameter")
    # sub_parser.add_argument(
    #     '-p', dest='p',
    #     default=2, type=int,
    #     help="set power (p) hyper-parameter")
    # sub_parser.add_argument(
    #     '--init-lr', dest='init_lr',
    #     default=3e-2, type=float,
    #     help="set initial learning rate")
    # sub_parser.add_argument(
    #     '--min-lr', dest='min_lr',
    #     default=3e-2, type=float,
    #     help="set minimum learning rate")
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
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.set_defaults(verbose=False)


def get_loss(loss: str) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss() if loss == 'cross_entropy' else \
        None


def main(args: APNamespace):
    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    global config

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
    with config_path.open() as f:
        config = yaml.load(f)
    print("Adas: Argument Parser Options")
    print("-"*45)
    print(f"    {'config':<20}: {args.config:<40}")
    print(f"    {'data':<20}: {str(Path(args.root) / args.data):<40}")
    print(f"    {'output':<20}: {str(Path(args.root) / args.output):<40}")
    print(f"    {'root':<20}: {args.root:<40}")
    print("\nAdas: LR Range Test: Config")
    print(f"    {'Key':<20} {'Value':<20}")
    print("-"*45)
    for k, v in config.items():
        print(f"    {k:<20} {v:<20}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"AdaS: Pytorch device is set to {device}")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if np.less(float(config['early_stop_threshold']), 0):
        print("AdaS: Notice: early stop will not be used as it was set to " +
              "{early_stop}, training till completion.")

    learning_rate = config['init_lr']
    if config['lr_scheduler'] == 'AdaS':
        filename = \
            f"stats_{config['optim_method']}_AdaS_LR_Range_test_" +\
            f"beta={config['beta']}_initlr={config['init_lr']}_" +\
            f"net={config['network']}_dataset={config['dataset']}.csv"
    else:
        filename = \
            f"stats_{config['optim_method']}_{config['lr_scheduler']}_" +\
            f"LR_Range_Test_initlr={config['init_lr']}" +\
            f"net={config['network']}_dataset={config['dataset']}.csv"
    if config['lr_scheduler'] == 'AdaS':
        xlsx_name = \
            f"{config['optim_method']}_AdaS_LR_Range_Test_" +\
            f"beta={config['beta']}_initlr={learning_rate}_" +\
            f"net={config['network']}_dataset=" +\
            f"{config['dataset']}.xlsx"
    else:
        xlsx_name = \
            f"{config['optim_method']}_{config['lr_scheduler']}" +\
            f"_LR_Range_Test_initlr={learning_rate}" +\
            f"net={config['network']}_dataset=" +\
            f"{config['dataset']}.xlsx"
    Profiler.filename = output_path / filename
    global performance_statistics
    performance_statistics = {}

    per_zero_thresh_low = 0.5  # float(config['per_zero_thresh'])
    per_zero_thresh_high = 0.9  # float(config['per_zero_thresh'])

    global counter
    counter = -1
    exit_counter = 0
    values = dict()
    per_non_zero_thresh = 0.93
    prev_validity_kg = True
    best_acc = 0.
    increase_factor = 10
    while True:
        counter += 1
        # Data
        # logging.info("Adas: Preparing Data")
        train_loader, test_loader = get_data(
            root=data_path,
            dataset=config['dataset'],
            mini_batch_size=config['mini_batch_size'])
        global net, metrics, adas

        # logging.info("AdaS: Building Model")
        net = get_net(config['network'], num_classes=10 if config['dataset'] ==
                      'CIFAR10' else 100 if config['dataset'] == 'CIFAR100'
                      else 1000 if config['dataset'] == 'ImageNet' else 10)
        if config['lr_scheduler'] == 'AdaS':
            adas = AdaS(parameters=list(net.parameters()),
                        beta=config['beta'],
                        zeta=config['zeta'],
                        init_lr=learning_rate,
                        min_lr=float(config['min_lr']),
                        p=config['p'])

        net = net.to(device)

        global criterion
        criterion = get_loss(config['loss'])

        optimizer, scheduler = get_optimizer_scheduler(
            net_parameters=net.parameters(),
            init_lr=learning_rate,
            optim_method=config['optim_method'],
            lr_scheduler=config['lr_scheduler'],
            train_loader_len=len(train_loader),
            max_epochs=int(config['max_epoch']))
        early_stop = EarlyStop(patience=int(config['early_stop_patience']),
                               threshold=float(config['early_stop_threshold']))

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        epochs = range(0, 3)
        metrics = Metrics(list(net.parameters()),
                          p=config['p'])
        for epoch in epochs:
            start_time = time.time()
            # print(f"AdaS: Epoch {epoch}/{epochs[-1]} Started.")
            train_loss, train_accuracy = epoch_iteration(
                train_loader, epoch, device, optimizer, scheduler)
            end_time = time.time()
            if config['lr_scheduler'] == 'StepLR':
                scheduler.step()
            test_loss, test_accuracy = test_main(
                test_loader, epoch, device)
            total_time = time.time()
            print(
                f"AdaS: LR Range Test " +
                f"Epoch {epoch}/{epochs[-1]} Ended | " +
                "Total Time: {:.3f}s | ".format(total_time - start_time) +
                "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (total_time - start_time) * (epochs[-1] - epoch)),
                "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                    train_loss,
                    train_accuracy) +
                "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(
                    test_loss,
                    test_accuracy))
            df = pd.DataFrame(data=performance_statistics)

            df.to_excel(str(output_path / xlsx_name))
            if early_stop(train_loss):
                print("AdaS: Early stop activated.")
                break
        per_in_S_zero = np.mean([np.mean(np.isclose(
            metric.input_channel_S,
            0).astype(np.float16)) for
            metric in metrics.historical_metrics])
        per_out_S_zero = np.mean([np.mean(np.isclose(
            metric.output_channel_S,
            0).astype(np.float16)) for
            metric in metrics.historical_metrics])
        # invalid_kg = any([np.any(np.array(metric.input_channel_S) > 1.) or
        #                   np.any(np.array(metric.output_channel_S) > 1.) for
        #                   metric in metrics.historical_metrics])
        if np.greater(test_accuracy, best_acc):
            best_acc = test_accuracy

        per_non_zero = 1. - np.mean([per_in_S_zero, per_out_S_zero])
        prev_learning_rate = learning_rate
        # if invalid_kg:
        #     learning_rate /= 10
        if np.less_equal(per_non_zero, per_non_zero_thresh):
            learning_rate *= increase_factor
        else:
            increase_factor = 1.3
            learning_rate /= 2
        # prev_validity_kg = not invalid_kg
        # if np.less(per_non_zero, 0.5):
        #     learning_rate *= 5
        # elif np.less(per_non_zero, 0.6):
        #     learning_rate *= 5
        # elif np.less(per_non_zero, 0.7):
        #     learning_rate *= 5
        # elif np.less(per_non_zero, 0.8):
        #     learning_rate *= 0.5
        # elif np.less(per_non_zero, 0.89):
        #     learning_rate *= 0.5
        # elif np.less(per_non_zero, 0.9):
        #     learning_rate *= 0.9
        # elif np.less(per_non_zero, 0.95):
        #     learning_rate *= 0.95
        # else:
        # values[prev_learning_rate] = per_non_zero
        # print(f"AdaS: LR Range Test Iteration {counter} | " +
        #       f"Learning Rate: {prev_learning_rate} | " +
        #       f"Percentage Non Zero: {per_non_zero} | " +
        #       f"Next Learning Rate: {learning_rate}")
        # with (output_path / 'lr_and_per_non_zero.txt').open('+w') as f:
        #     for k, v in values.items():
        #         f.write(f"LR: {k} | Perc. Non Zero: {v}\n")
        # break
        values[prev_learning_rate] = [per_non_zero, test_accuracy]
        print(f"AdaS: LR Range Test Iteration {counter} | " +
              f"Learning Rate: {prev_learning_rate} | " +
              # f"Valid KG: {not invalid_kg} | " +
              f"Percentage Non Zero: {per_non_zero} | " +
              f"Next Learning Rate: {learning_rate}")
        with (output_path / 'lr_and_per_non_zero.txt').open('+w') as f:
            for lr, (pnz, acc) in values.items():
                f.write(f"LR: {lr} | Perc. Non Zero: {pnz}\n")
        if counter >= 15:
            print("AdaS: LR Range Test did 15 iterations, breaking")
            break
        print("AdaS: LR Range Test Complete")
        print(f"    {'LR':<20} {'Perc. Non Zero':<20} {'Test Accuracy':<20}")
        for lr, (pnz, acc) in values.items():
            print(f"    {lr:<20} {pnz:<20} {acc:<20}")


def test_main(test_loader, epoch: int, device) -> Tuple[float, float]:
    global performance_statistics, net, criterion, counter
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    performance_statistics[f'lrrt_{counter}_acc_epoch_' +
                           str(epoch)] = acc / 100
    return test_loss / (batch_idx + 1), acc


def epoch_iteration(train_loader, epoch: int,
                    device, optimizer, scheduler) -> Tuple[float, float]:
    # logging.info(f"Adas: Train: Epoch: {epoch}")
    global net, performance_statistics, metrics, adas, config, counter
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    """train CNN architecture"""
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if config['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + batch_idx / len(train_loader))
        optimizer.zero_grad()
        if config['optim_method'] == 'SLS':
            def closure():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                return loss, outputs
            loss, outputs = optimizer.step(closure=closure)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if adas is not None:
                optimizer.step(metrics.layers_index_todo,
                               adas.lr_vector)
            elif config['optim_method'] == 'SPS':
                optimizer.step(loss=loss)
            else:
                optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if config['lr_scheduler'] == 'OneCycleLR':
            scheduler.step()

        # progress_bar(batch_idx, len(train_loader),
        #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1),
        #                  100. * correct / total, correct, total))
    performance_statistics[f'lrrt_{counter}_Train_loss_epoch_' +
                           str(epoch)] = train_loss / (batch_idx + 1)

    io_metrics = metrics.evaluate(epoch)
    performance_statistics[f'lrrt_{counter}_in_S_epoch_' +
                           str(epoch)] = io_metrics.input_channel_S
    performance_statistics[f'lrrt_{counter}_out_S_epoch_' +
                           str(epoch)] = io_metrics.output_channel_S
    performance_statistics[f'lrrt_{counter}_fc_S_epoch_' +
                           str(epoch)] = io_metrics.fc_S
    performance_statistics[f'lrrt_{counter}_in_rank_epoch_' +
                           str(epoch)] = io_metrics.input_channel_rank
    performance_statistics[f'lrrt_{counter}_out_rank_epoch_' +
                           str(epoch)] = io_metrics.output_channel_rank
    performance_statistics[f'lrrt_{counter}_fc_rank_epoch_' +
                           str(epoch)] = io_metrics.fc_rank
    performance_statistics[f'lrrt_{counter}_in_condition_epoch_' +
                           str(epoch)] = io_metrics.input_channel_condition
    performance_statistics[f'lrrt_{counter}_out_condition_epoch_' +
                           str(epoch)] = io_metrics.output_channel_condition
    if adas is not None:
        lrmetrics = adas.step(epoch, metrics)
        performance_statistics[f'lrrt_{counter}_rank_velocity_epoch_' +
                               str(epoch)] = lrmetrics.rank_velocity
        performance_statistics[f'lrrt_{counter}_learning_rate_epoch_' +
                               str(epoch)] = lrmetrics.r_conv
    else:
        if config['optim_method'] == 'SLS' or config['optim_method'] == 'SPS':
            performance_statistics[f'lrr_{counter}_learning_rate_epoch_' +
                                   str(epoch)] = optimizer.state['step_size']
        else:
            performance_statistics[f'lrrt_{counter}_learning_rate_epoch_' +
                                   str(epoch)] = optimizer.param_groups[0]['lr']
    return train_loss / (batch_idx + 1), 100. * correct / total


if __name__ == "__main__":
    ...
