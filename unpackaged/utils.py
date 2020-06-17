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

Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
from typing import Dict, Union, List

import pstats


def safe_float_cast(str_number: str) -> float:
    try:
        number = float(str_number)
    except ValueError:
        number = float('nan')
    return number


def pstats_to_dict(stats: pstats.Stats) -> List[Dict[str, Union[str, float]]]:
    formatted_stats = list()
    stats = 'ncalls'+stats.split('ncalls')[-1]
    stats = [line.rstrip().split(None, 5) for line in
             stats.split('\n')]
    for stat in stats[1:]:
        stats_dict = dict()
        if len(stat) >= 5:
            stats_dict['n_calls'] = stat[0]
            stats_dict['tot_time'] = stat[1]
            stats_dict['per_call1'] = stat[2]
            stats_dict['cum_time'] = stat[3]
            stats_dict['per_call2'] = stat[4]
            name = stat[5].split(':')
            stats_dict['name'] = \
                f"{name[0].split('/')[-1]}_line(function)_{name[1]}"
            formatted_stats.append(stats_dict)
    return formatted_stats
