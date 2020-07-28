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

from .alexnet import alexnet as AlexNet
from .densenet import densenet201 as DenseNet201, densenet169 as DenseNet169, \
    densenet161 as DenseNet161, densenet121 as DenseNet121
from .googlenet import googlenet as GoogLeNet
from .inception import inception_v3 as InceptionV3
from .mnasnet import mnasnet0_5 as MNASNet_0_5, mnasnet0_75 as MNASNet_0_75, \
    mnasnet1_0 as MNASNet_1, mnasnet1_3 as MNASNet_1_3
from .mobilenet import mobilenet_v2 as MobileNetV2
from .resnet import resnet18 as ResNet18, resnet34 as ResNet34, \
    resnet50 as ResNet50, resnet101 as ResNet101, resnet152 as ResNet152, \
    resnext50_32x4d as ResNext50, resnext101_32x8d as ResNext101, \
    wide_resnet50_2 as WideResNet50, wide_resnet101_2 as WideResNet101
from .shufflenetv2 import shufflenet_v2_x0_5 as ShuffleNetV2_0_5, \
    shufflenet_v2_x1_0 as ShuffleNetV2_1, \
    shufflenet_v2_x1_5 as ShuffleNetV2_1_5, \
    shufflenet_v2_x2_0 as ShuffleNetV2_2
from .squeezenet import squeezenet1_0 as SqueezeNet_1, \
    squeezenet1_1 as SqueezeNet_1_1
from .vgg import vgg11 as VGG11, vgg11_bn as VGG11_BN, \
    vgg13 as VGG13, vgg13_bn as VGG13_BN, vgg16 as VGG16, \
    vgg16_bn as VGG16_BN, vgg19 as VGG19, vgg19_bn as VGG19_BN


def get_net(network: str,
            num_classes: int) -> torch.nn.Module:
    return AlexNet(num_classes=num_classes) if network == 'AlexNet' else\
        DenseNet201(num_classes=num_classes) if network == 'DenseNet201' else\
        DenseNet169(num_classes=num_classes) if network == 'DenseNet169' else\
        DenseNet161(num_classes=num_classes) if network == 'DenseNet161' else\
        DenseNet121(num_classes=num_classes) if network == 'DenseNet121' else\
        GoogLeNet(num_classes=num_classes) if network == 'GoogLeNet' else\
        InceptionV3(num_classes=num_classes) if network == 'InceptionV3' else\
        MNASNet_0_5(num_classes=num_classes) if network == 'MNASNet_0_5' else\
        MNASNet_0_75(
        num_classes=num_classes) if network == 'MNASNet_0_75' else\
        MNASNet_1(num_classes=num_classes) if network == 'MNASNet_1' else\
        MNASNet_1_3(num_classes=num_classes) if network == 'MNASNet_1_3' else\
        MobileNetV2(num_classes=num_classes) if network == 'MobileNetV2' else\
        ResNet18(num_classes=num_classes) if network == 'ResNet18' else\
        ResNet34(num_classes=num_classes) if network == 'ResNet34' else\
        ResNet50(num_classes=num_classes) if network == 'ResNet50' else\
        ResNet101(num_classes=num_classes) if network == 'ResNet101' else\
        ResNet152(num_classes=num_classes) if network == 'ResNet152' else\
        ResNext50(num_classes=num_classes) if network == 'ResNext50' else\
        ResNext101(num_classes=num_classes) if network == 'ResNext101' else\
        WideResNet50(
        num_classes=num_classes) if network == 'WideResNet50' else\
        WideResNet101(
        num_classes=num_classes) if network == 'WideResNet101' else\
        ShuffleNetV2_0_5(
        num_classes=num_classes) if network == 'ShuffleNetV2_0_5' else\
        ShuffleNetV2_1(
        num_classes=num_classes) if network == 'ShuffleNetV2_1' else\
        ShuffleNetV2_1_5(
        num_classes=num_classes) if network == 'ShuffleNetV2_1_5' else\
        ShuffleNetV2_2(
        num_classes=num_classes) if network == 'ShuffleNetV2_2' else\
        SqueezeNet_1(
        num_classes=num_classes) if network == 'SqueezeNet_1' else\
        SqueezeNet_1_1(
        num_classes=num_classes) if network == 'SqueezeNet_1_1' else\
        VGG11(num_classes=num_classes) if network == 'VGG11' else\
        VGG11_BN(num_classes=num_classes) if network == 'VGG11_BN' else\
        VGG13(num_classes=num_classes) if network == 'VGG13' else\
        VGG13_BN(num_classes=num_classes) if network == 'VGG13_BN' else\
        VGG16(num_classes=num_classes) if network == 'VGG16' else\
        VGG16_BN(num_classes=num_classes) if network == 'VGG16_BN' else\
        VGG19(num_classes=num_classes) if network == 'VGG19' else\
        VGG19_BN(num_classes=num_classes) if network == 'VGG19_BN' else None
