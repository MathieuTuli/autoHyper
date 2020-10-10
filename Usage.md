# Usage #
#### Config File ####
In the following sections we list the configuration options available to the user. Note that we also classify the configuration options into the following categories:
- Application Specific
  - dataset
  - network
  - optimizer
  - scheduler
- Suggested Tune
   - init_lr
   - early_stop_threshold
   - optimizer_kwargs
   - scheduler_kwargs
- Suggested Default
   - n_trials
   - max_epochs
   - mini_batch_size
   - num_workers
   - loss
   - early_stop_patience
   - p

The **Suggested Default** parameters are ones we have preset and suggest not be altered too much. Naturally, the user may change them at their discretion.

The **Suggested Tune** parameters are highly recommended to be tuned, and are very application specific.

The **Application Specific** parameters then are simply ones that the user must change to do what they want (what dataset, model, learning algorithm, etc.)


#### Available Datasets for Training ####
---
**yaml identifier: dataset**
Currently the following datasets are supported:
- CIFAR10
- CIFAR100
- ImageNet

#### Available Models for Training ####

---
**yaml identifier: network**
All models used can be found in [src/autohyper/models](src/adas/models). They are a combination of [PyTorch](https://github.com/pytorch/pytorch) and [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) models.
- AlexNet
- DenseNet201 | DenseNet169 | DenseNet161 | DenseNet121 | DenseNet121CIFAR
- GoogLeNet
- InceptionV3
- MNASNet_0_5 | MNASNet_0_75 | MNASNet_1 | MNASNet_1_3
- MobileNetV2
- ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 | ResNext50 | ResNext101 | ResNet18CIFAR | ResNet34CIFAR
- ResNeXtCIFAR
- WideResNet50 | WideResNet101
- ShuffleNetV2_0_5 | ShuffleNetV2_1 | ShuffleNetV2_1_5 | ShuffleNetV2_2
- SqueezeNet_1 | SqueezeNet_1_1
- VGG11 | VGG11_BN | VGG13 | VGG13_BN | VGG16 | VGG16_BN | VGG19 | VGG19_BN
- VGG16CIFAR
- EfficientNetB4 | EfficientNetB0CIFAR


#### Optimizer Method ####

---
**yaml identifier: optimizer**

Options:
- SGD
- NAG
- AdaM
- AdaGrad
- RMSProp
- AdaDelta
- AdaBound
- AMSBound
- AdaMax
- AdaMod
- AdaShift
- NAdam
- NosAdam
- NovoGrad
- PAdam
- RAdam
- SPS
- SLS
- LaProp
- LearningRateDropout

#### Learning Rate Scheduler ####

---
**yaml identifier: scheduler**

Options:
- AdaS (Note that `SGD` must be specified as the `optimizer`)
- StepLR
- CosineAnnealingWarmRestarts
- OneCycleLR

#### Initial Learning Rate ####

---
**yaml identifier: init_lr**

Initial learning rate for the optimizer method. Note that specifying 'auto' will run the autoHyper algorithm to determine the optimal initial learning rate.

#### Early Stopping Threshold ####

---
**yaml identifier: early_stop_threshold**

***Note that early stopping should only be used for the SGD with AdaS algorithm. As per the paper, AdaS provides the ability to monitor simply training loss and be confident that a low training loss leads to a high test accuracy/low testing loss. Hence, we only use early stop for SGD with AdaS, monitoring the training loss, and do not recommend its use otherwise. ***

The threshold for early stopping. The early stopping criterion operates by keeping track of the best loss seen to date, and evaluates the current loss against the best loss by doing `current_loss - best_loss`. If this value is **greater than** the early stopping threshold, a counter begins. If this evaluation is true for `early_stop_patience` (see below) amount of epochs, then early stopping is activated.

To deactivate early_stopping, set this value to `-1`.

#### Optimizer Arguments ####

---
**yaml identifier: optimizer_kwargs**

Specific arguments to pass to the selected optimizer. Expecting a dictionary, where keys are the exact argument names. There are certain required arguments for certain optimizers, which can be seen listed in [src/autohyper/optim/__init__.py](src/autohyper/optim/__init__.py). If not passing any arguments, ensure an empty list is the value.

#### Scheduler Arguments ####

---
**yaml identifier: scheduler_kwargs**

Same as above, but for scheduler argument.

#### Number of Training Trials ####

---
**yaml identifier: n_trials**

Number of full training cycles


#### Max Epochs ####

---
**yaml identifier: max_epoch**

Maximum number of epochs for one trial

#### Mini-Batch Size ####

---
**yaml identifier: mini_batch_size**

Size of mini-batch for one epoch

#### Early Stopping Patience ####

---
**yaml identifier: early_stop_patience**

Patience window for early stopping.

#### Power ####

---
**yaml identifier: p**

Power value for computing knowledge-gain. Can either be `1` or `2`.
