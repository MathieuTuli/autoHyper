import torch
import torch.nn as nn


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, C, H // self.bs, self.bs, W // self.bs,
                      self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        # (N, bs, bs, C, H//bs, W//bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.reshape(N, C * (self.bs ** 2), H // self.bs, W //
                      self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        # (N, C, H//bs, bs, W//bs, bs)
        x = x.reshape(N, C, H // 4, 4, W // 4, 4)
        # (N, bs, bs, C, H//bs, W//bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.reshape(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, no_jit=False):
        super().__init__()
        if not no_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, self.bs, self.bs, C // (self.bs ** 2),
                      H, W)  # (N, bs, bs, C//bs^2, H, W)
        # (N, C//bs^2, H, bs, W, bs)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.reshape(N, C // (self.bs ** 2), H * self.bs, W *
                      self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x
