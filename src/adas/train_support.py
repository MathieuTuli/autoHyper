from typing import Tuple

import time

import pandas as pd

from . import global_vars as GLOBALS
from .profiler import Profiler
from .AdaS import AdaS
from .test import test_main
from .optim.sls import SLS
from .optim.sps import SPS


def run_epochs(trial, epochs, train_loader, test_loader,
               device, optimizer, scheduler, output_path):
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
    xlsx_path = str(output_path / xlsx_name)
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

        df.to_excel(xlsx_path)
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
        # print(f'{batch_idx} / {len(train_loader)}')
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
