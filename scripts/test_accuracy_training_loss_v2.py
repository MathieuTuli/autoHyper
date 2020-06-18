"""
author: Mathieu Tuli
source: https://github.com/MathieuTuli/AdaS

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

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = ArgumentParser()
parser.add_argument('--files', dest='files',
                    type=str, nargs="+",
                    help="list of XLSX file paths. " +
                    "If you wish to perform for " +
                    "multiple trials (and average over them)," +
                    "place each trial file in the " +
                    "same directory and pass the directory path")
args = parser.parse_args()
acc_min = 0.6
acc_max = 1.0
loss_min = 5e-4
loss_max = 10
color_codes = ['steelblue', 'b', 'g', 'c', 'm', 'y', 'orange']
line_style = ['-', '-', '-', '-', '-', '--',
              '--', '--', '--', '--', '--', '--']
plt.figure(1, figsize=(5, 5))
for i, filename in enumerate(args.files):
    label = f"{filename.split('/')[-1].split('_')[0]}_{filename.split('/')[-1].split('_')[1]}"
    total_acc_data_vec = list()
    df = pd.read_excel(filename)
    df = df.T
    if "adas" in filename.split('/')[-1].lower():
        acc_data_vec = np.asarray(df.iloc[12::12, 1])
    else:
        acc_data_vec = np.asarray(df.iloc[10::10, 1])
        total_acc_data_vec.append(acc_data_vec)
    plt.plot(np.array(range(1, len(acc_data_vec) + 1)),
             acc_data_vec,
             c=color_codes[i],
             linestyle=line_style[i], label=label)
plt.ylim((acc_min,
          acc_max))

plt.xlim((1, 250))
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Epoch - (t)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={"size": 9}, loc="upper right",
           bbox_to_anchor=(0.98, 0.98), borderaxespad=0., ncol=2)
plt.grid(True)
export_name = 'accuracy_comparison.png'
plt.savefig(export_name, dpi=300, bbox_inches='tight')
plt.close()

plt.figure(1, figsize=(5, 5))
for i, filename in enumerate(args.files):
    label = f"{filename.split('/')[-1].split('_')[0]}_{filename.split('/')[-1].split('_')[1]}"
    total_err_vec = list()
    df = pd.read_excel(filename)
    df = df.T
    if "adas" in filename.split('/')[-1].lower():
        error_data_vec = np.asarray(df.iloc[1::12, 1])
    else:
        error_data_vec = np.asarray(df.iloc[1::10, 1])

    plt.plot(np.array(range(1, len(error_data_vec) + 1)),
             error_data_vec,
             c=color_codes[i],
             linestyle=line_style[i], label=label)
plt.ylim((loss_min,
          loss_max))
plt.xlim((1, 250))
plt.yscale('log', basey=10)
plt.ylabel('Training Loss', fontsize=16)
plt.xlabel('Epoch - (t)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={"size": 9}, loc="upper right",
           bbox_to_anchor=(0.98, 0.98), borderaxespad=0., ncol=2)
plt.grid(True)
export_name = f'loss_comparison.png'
plt.savefig(export_name, dpi=300, bbox_inches='tight')
plt.close()
