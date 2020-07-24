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
from typing import Tuple

import torch

from .vgg import VGG
from .dpn import DPN92
# from .lenet import *
from .senet import SENet18
# from .pnasnet import *
from .densenet import densenet121
from .googlenet import GoogLeNet
from .shufflenet import ShuffleNetG2
from .shufflenetv2 import ShuffleNetV2
from .resnet import resnet34 as ResNet34
from .resnext import ResNeXt29_2x64d
from .preact_resnet import PreActResNet18
from .mobilenet import mobilenet_v2 as MobileNet
# from .mobilenetv2 import MobileNetV2
from .efficientnet import EfficientNetB0


def get_net(network: str,
            num_classes: int, input_size: Tuple[int, int]) -> torch.nn.Module:
    return VGG(
        'VGG16', num_classes=num_classes,
        input_size=input_size) if network == 'VGG16' else \
        ResNet34(
        num_classes=num_classes,
        input_size=input_size) if network == 'ResNet34' else \
        PreActResNet18(
        num_classes=num_classes,
        input_size=input_size) if network == 'PreActResNet18' else \
        GoogLeNet(
        num_classes=num_classes,
        input_size=input_size) if network == 'GoogLeNet' else \
        densenet121(
        num_classes=num_classes,
        input_size=input_size) if network == 'densenet_cifar' else \
        ResNeXt29_2x64d(
        num_classes=num_classes,
        input_size=input_size) if network == 'ResNeXt29_2x64d' else \
        MobileNet(
        num_classes=num_classes,
        input_size=input_size) if network == 'MobileNet' else \
        DPN92(num_classes=num_classes,
              input_size=input_size) if network == 'DPN92' else \
        ShuffleNetG2(num_classes=num_classes,
                     input_size=input_size) if network == 'ShuffleNetG2' else \
        SENet18(num_classes=num_classes,
                input_size=input_size) if network == 'SENet18' else \
        ShuffleNetV2(1, num_classes=num_classes,
                     input_size=input_size) if network == 'ShuffleNetV2' else \
        EfficientNetB0(
            num_classes=num_classes) if network == 'EfficientNetB0' else None
