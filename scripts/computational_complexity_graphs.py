"""
author: Mathieu Tuli
source: https://github.com/MathieuTuli/AdaS

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
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = ArgumentParser(description=__doc__)
parser.add_argument(
    '--file', dest='file', required=True,
    help='CSV file path containing the computational complexity data')

args = parser.parse_args()

if __name__ == "__main__":
    if not Path(args.file).exists():
        print(f"{args.file} does not exist")
        raise ValueError
    df = pd.read_csv(args.file)  # , index_col='epoch')
    num_plots = int(2 + ((len(df.columns) - 3) / 4))
    data = dict()
    fig, axs = plt.subplots(num_plots)
    fig.set_size_inches(10, 10)
    assert('epoch' in df.columns)
    x_data = np.array(df['epoch'])
    per_calls_idx = 2
    for k, v in df.items():
        if 'ram' in k:
            axs[0].plot(x_data, np.array(v), c='r')
            axs[0].set_xlabel('Epochs', fontsize=8)
            axs[0].set_ylabel('Memory (GB)', fontsize=8)
            axs[0].set_title('RAM Used', fontsize=8)
        if 'gpu' in k:
            axs[1].plot(x_data, np.array(v), c='r')
            axs[1].set_xlabel('Epochs', fontsize=8)
            axs[1].set_ylabel('Memory (GB)', fontsize=8)
            axs[1].set_title('GPU Memory Used', fontsize=8)
        if 'per_call2' in k:
            axs[per_calls_idx].plot(x_data, np.array(v), c='g')
            axs[per_calls_idx].set_xlabel('Epochs', fontsize=8)
            axs[per_calls_idx].set_ylabel('Call Time (s)', fontsize=8)
            axs[per_calls_idx].set_title(k, fontsize=8)
            per_calls_idx += 1
    plt.tight_layout()
    filename = Path(args.file)
    output = Path(args.file).parent / 'computational_complexity_stats.png'
    plt.savefig(str(output),
                dpi=100, bbox_inches='tight')
    print(f"File saved to {output}")
