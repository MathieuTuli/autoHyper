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
from pathlib import Path
from typing import Dict
from autohyper import optimize, LowRankMetrics, HyperParameters
from torchvision import datasets, transforms
from torch.optim import Adam
from gutils import init_logger

import torchvision.models as models
import numpy as np
import torch


def main():
    # indicate which hyper-parameters to optimize
    dataset = torch.utils.data.DataLoader(
        datasets.CIFAR10('.', download=True, transform=transforms.ToTensor()),
        batch_size=128)

    def epoch_trainer(hyper_parameters: Dict[str, float],
                      epochs) -> LowRankMetrics:
        # update model/optimizer parameters based on values in @argument:
        #     hyper_parameters
        print('Run epochs:', hyper_parameters)
        model = models.resnet18()
        model.train()
        model = model.cuda()
        metrics = LowRankMetrics(list(model.parameters()))
        optimizer = Adam(model.parameters(),
                         lr=hyper_parameters['lr'],
                         weight_decay=hyper_parameters['weight_decay'],)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        accs = list()
        for epoch in epochs:
            for inputs, targets in dataset:
                inputs = inputs.cuda()
                targets = targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                accs.append(accuracy(outputs, targets)[0].item())
            # run epoch training...
            # at every epoch, evaluate low_rank metrics
            print(f"Epoch {epoch} | Loss {np.mean(accs)}")
            metrics.evaluate()
        return metrics

    hyper_parameters = HyperParameters(lr=True, weight_decay=True)
    final_hp = optimize(epoch_trainer=epoch_trainer,
                        hyper_parameters=hyper_parameters)
    final_hyper_parameters_dict = final_hp.final()
    # do your final training will optimized hyper parameters

    epoch_trainer(final_hyper_parameters_dict, epochs=range(250))


def accuracy(outputs, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    logger = init_logger(Path('logs'))
    main()
