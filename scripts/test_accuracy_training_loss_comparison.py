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
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os

LIMIT_EPOCHS = False
FONTSIZE = 9
evaluation_directory = '/home/mat/playgrounds/new-test'
evaluation_directory = '/home/mat/archive/training/AdaS/adaptive-learning-survey/005-trial/'
evaluation_directory = '/home/mat/archive/training/AdaS/lr-range-test/correctness-check/'
evaluation_directory = '/home/mat/archive/training/AdaS/lr-range-test/autolr'
evaluation_directory = '/home/mat/archive/training/AdaS/adaptive-learning-survey/paper-based-trial/iteration-2/'
evaluation_directory = '/home/mat/work/U-of-T/summer-research/mml/lr-range-feature/draft/versus'
evaluation_directory = '/home/mat/playgrounds/new/lr-range-test/'
evaluation_directory = '/home/mat/archive/training/AdaS/new-model-check'
evaluation_directory = '/home/mat/playgrounds/lambda/resnext50-cifar10'
evaluation_directory = '/home/mat/playgrounds/t'
evaluation_directory = '/home/mat/archive/training/AdaS/lr-range-test/ema-test/lambda-server-results/ema-test/vgg16-cifar10/'
evaluation_directory = '/home/mat/archive/training/AdaS/lr-range-test/draft-experiments/iteration-2-regular/vgg16-cifar10'
evaluation_directory = '/home/mat/playgrounds/lambda-output/ema-test/vgg16-cifar10'
evaluation_directory = '/home/mat/playgrounds/cumprod08/iteration-4-final/resnet34-cifar100/'
evaluation_directory = '/home/mat/work/U-of-T/summer-research/mml/lr-range-feature/cumprod-08-results/iteration-4-final'
evaluation_directory = '/home/mat/playgrounds/iteration4/iteration-4-final/resnext-cifar10/'
evaluation_directory = '/home/mat/playgrounds/steplr-auto//iteration-6-step-95//resnet34-cifar100/'

# for net in Path(evaluation_directory).iterdir():
for net in ['resnext']:
    print(net)
    EPOCHS = 250
    optimizers = list()
    global optimizer
    l_sorted_files = sorted(Path(evaluation_directory).iterdir())
    sorted_files = list()
    for optimizer_folder in l_sorted_files:
        if 'AdaGrad' in str(optimizer_folder) or 'RMSProp' in str(optimizer_folder) or 'SLS' in str(optimizer_folder):
            ...
        if optimizer_folder.is_dir():
            if (optimizer_folder / '.output').exists():
                for f in (optimizer_folder / '.output').iterdir():
                    if 'lr-' in f.name and f.is_dir() and 'auto' not in f.name:
                        sorted_files.append(f)
                        folder = str(optimizer_folder).split('/')[-1]
                        optimizers.append(
                            folder + f'-{float(f.name.replace("lr-", "")):.5f}')
            else:
                print(f"{optimizer_folder} has no files")

# fig = plt.figure()
# x = np.linspace(np.min(learning_rates), np.max(learning_rates),
#                 len(learning_rates))
    x = np.array(optimizers)
    y = np.arange(EPOCHS)
    train_acc = np.zeros((len(y), len(x)))
    train_loss = np.zeros((len(y), len(x)))
    test_acc = np.zeros((len(y), len(x)))
    test_loss = np.zeros((len(y), len(x)))
    test_std = np.zeros((len(y), len(x)))
    i = -1
    for s, optimizer_folder in enumerate(sorted_files):
        files = list()
        if not optimizer_folder.is_dir():
            continue
        i += 1
        # for excel_file in (optimizer_folder / '.adas-output').iterdir():
        #     # if 'lr-' in str(d.name):
        #     # for excel_file in d.iterdir():
        #     if '.csv' == excel_file.suffix:
        #         continue
        #     files.append(str(excel_file))
        # for d in (optimizer_folder / '.output').iterdir():
        for excel_file in (optimizer_folder).iterdir():
            if '.csv' == excel_file.suffix:
                continue
            files.append(str(excel_file))
        train_acc_data = np.empty((EPOCHS, len(files)))
        train_acc_data[:] = np.nan
        train_loss_data = np.empty((EPOCHS, len(files)))
        train_loss_data[:] = np.nan
        test_acc_data = np.empty((EPOCHS, len(files)))
        test_acc_data[:] = np.nan
        test_loss_data = np.empty((EPOCHS, len(files)))
        test_loss_data[:] = np.nan
        for j, f in enumerate(files):
            df = pd.read_excel(str(f)).T
            adas_offset = 1 if 'AdaS' in str(f).split('/')[-1] else 0
            adas_offset = 0 if 'AdaShift' in str(
                f).split('/')[-1] else adas_offset

            train_acc_vec = np.asarray(
                df.iloc[1::13 + adas_offset, :])[:EPOCHS, :]
            train_loss_vec = np.asarray(
                df.iloc[2::13 + adas_offset, :])[:EPOCHS, :]
            test_acc_vec = np.asarray(
                df.iloc[12 + adas_offset::13 + adas_offset, :])[:EPOCHS, :]
            test_loss_vec = np.asarray(
                df.iloc[13 + adas_offset::13 + adas_offset, :])[:EPOCHS, :]
            train_acc_data[:len(train_acc_vec), j] = train_acc_vec[:, 0]
            train_loss_data[:len(train_loss_vec), j] = train_loss_vec[:, 0]
            test_acc_data[:len(test_acc_vec), j] = test_acc_vec[:, 0]
            test_loss_data[:len(test_loss_vec), j] = test_loss_vec[:, 0]
        print('--')
        print(optimizers[s])
        # print(train_acc_data.mean(1)[-1])
        # print(train_acc_data.std(1)[-1])
        # print(train_loss_data.mean(1)[-1])
        # print(train_loss_data.std(1)[-1])
        nan_locs = np.where(np.isnan(test_acc_data))
        len_ = len(test_acc_data) if len(
            nan_locs[1]) == 0 else nan_locs[0][0]
        print(f"ACC at 250: {100*test_acc_data[:len_, :].mean(1)[-1]}%")
        print(f"Max ACC : {100*np.max(test_acc_data[:len_, :].mean(1))}%")
        std = 100*test_acc_data[:len_, :].std(1)[-1]
        print(f"STD: {std}%")
        std = test_acc_data.std(1)
        # print(test_loss_data.mean(1)[-1])
        # print(test_loss_data.std(1)[-1])
        train_acc[:, i] = train_acc_data.mean(1)
        train_loss[:, i] = train_loss_data.mean(1)
        test_acc[:, i] = test_acc_data.mean(1)
        test_loss[:, i] = test_loss_data.mean(1)
        test_std[:, i] = std

    if LIMIT_EPOCHS:
        y = y[:10]

    color_codes = [(1, 0, 0), (0.8, 0, 0), (0.6, 0, 0), (0.4, 0, 0),
                   (0.3, 0, 0), 'steelblue', 'b', 'g', 'c', 'm', 'y', 'orange']
    color_codes = list(np.array(
        [(48, 252, 3), (16, 77, 3), (0, 255, 255), (0, 74, 74),
         (255, 132, 0), (112, 58, 0), (255, 10, 35), (120, 0, 12),
         (223, 13, 255), (69, 3, 79), (232, 218, 23), (110, 102, 1)])/255.)
    random.seed(1)
    random.shuffle(color_codes)
    line_style = ['-', '--', 'dotted']
    acc_min = [0.82, 0.85, 0.60, 0.63]
    acc_max = [0.945, 0.96, 0.74, 0.78]
    loss_min = [5e-4, 5e-4, 1e-3, 1e-3]
    loss_max = [10, 10, 10, 10]
    for (name, data) in [('train_acc', train_acc), ('train_loss', train_loss),
                         ('test_acc', test_acc), ('test_loss', test_loss)]:
        for i, optimizer in enumerate(x):
            idx = i % len(color_codes)
            plt.plot(np.array(range(1, data.shape[0] + 1)),
                     data[:, i], linestyle=line_style[int(
                         i / len(color_codes))],
                     color=color_codes[idx], zorder=0)
            if name == 'test_acc' and False:
                plt.fill_between(np.array(range(
                    1, data.shape[0] + 1)), data[:, i] - std, data[:, i] + std, color=color_codes[idx], alpha=.1, zorder=100)
        plt.grid(b=True, which='both', axis='both')
        plt.gca().legend([f'{optimizer}' for optimizer in x],
                         prop={"size": 9}, loc="lower right",
                         bbox_to_anchor=(
            0.98, 0.02), borderaxespad=0., ncol=3)
        plt.xlabel('Epoch', size=9)
        plt.xlim((1, EPOCHS))
        if 'train_loss' == name or 'test_loss' == name:
            # plt.ylim(0, 1)
            plt.yscale('log', basey=10)
        elif 'train_acc' == name:
            plt.ylim(0.82, 1.)
        else:
            if 'cifar100' in str(net).lower() and 'efficientnet' in str(net).lower():
                plt.ylim(0.4, 0.8)
                plt.ylim(0.59, 0.8)
        plt.ylim(0.5, 0.96)
        plt.ylabel(f'{name}', size=9)
        # plt.savefig(f'comparison_{net.name}_{name}.png',
        #             dpi=300, bbox_inces='tight')
        plt.savefig(f'comparison_{net}_{name}.png',
                    dpi=300, bbox_inces='tight')
        plt.close()
