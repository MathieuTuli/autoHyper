"""
MIT License

Copyright (c) 2020

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
from typing import NamedTuple, List, Union, Dict
from dataclasses import dataclass
from enum import Enum

# import numpy as np

import logging


class LogLevel(Enum):
    '''
    What the stdlib did not provide!
    '''
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __str__(self):
        return self.name


class LayerType(Enum):
    CONV = 1
    FC = 2
    NON_CONV = 3


class IOMetrics(NamedTuple):
    input_channel_rank: List[float]
    input_channel_S: List[float]
    input_channel_condition: List[float]
    output_channel_rank: List[float]
    output_channel_S: List[float]
    output_channel_condition: List[float]
    fc_S: float
    fc_rank: float


class LRMetrics(NamedTuple):
    rank_velocity: List[float]
    r_conv: List[float]


class Statistics(NamedTuple):
    ram: float
    gpu_mem: float
    epoch_time: float
    step_time: float


@dataclass
class HyperParameter():
    current: Union[float, int] = None
    minimum: float = None
    scale: float = None
    stop: bool = False
    count: int = 0


@dataclass
class HyperParameters():
    lr: bool = False
    weight_decay: bool = False
    parameters = {
        'lr': HyperParameter(current=1e-4, scale=1.45, minimum=1e-6),
        'weight_decay': HyperParameter(current=0, scale=2., minimum=0)
    }

    def __len__(self) -> int:
        return len(self.parameters)

    def __getitem__(self, key: str) -> HyperParameter:
        return self.parameters[key]

    def __setitem__(self, key: str, value: HyperParameter):
        if not isinstance(key, str):
            raise ValueError("Key must be of type 'str'")
        if not isinstance(value, HyperParameter):
            raise ValueError("Value must be of type 'HyperParameter'")
        self.parameters[key] = value

    def __iter__(self):
        return iter(self.parameters)

    def keys(self):
        return self.parameters.keys()

    def values(self):
        return self.parameters.values()

    def items(self):
        return self.parameters.items()

    def final(self) -> Dict[str, HyperParameter]:
        return {param: val.current for param, val in self.parameters.items()}
