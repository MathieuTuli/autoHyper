# Requirements #
In order to satisfy `torch==1.6.0` the following (soft) Nvidia requirements need to be met:
- CUDA Version: `CUDA 10.2`
- CUDA Driver Version: `r440`
- CUDNN Version: `7.6.4-7.6.5`
For more information, refer to the [cudnn-support-matrix](https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html).

Refer to the [PyTorch installation guide](https://pytorch.org/) for information how to install PyTorch. We do not guarantee proper function of this code using different versions of PyTorch or CUDA.

#### Hardware ####
- GPU
  - At least 8 GB of GPU memory is required
- Memory
  - At least 8 GB RAM is required

Naturally, the memory requirements is scaled relative to current dataset being used and mini-batch sizes. ImageNet experiments for example will need the full 8 GB perhaps, whereas CIFAR10 experiments might need much less.
