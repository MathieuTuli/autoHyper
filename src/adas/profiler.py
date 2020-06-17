from typing import Tuple
from pstats import SortKey
from pathlib import Path

import cProfile
import pstats
import io

from memory_profiler import memory_usage

from .gpu import GPU


class Profiler:
    gpu_id: int = 0
    root: Path = Path('.')

    def __init__(self, function):
        self.epoch = 0
        self.gpu = GPU(Profiler.gpu_id)
        self.stream = (Profiler.root / 'profiler.log').open('w+')
        self.pr = cProfile.Profile()
        self.function = function

    def __call__(self, train_loader, epoch: int,
                 device, optimizer, scheduler) -> Tuple[float, float]:
        self.gpu.update()
        self.pr.enable()
        result = memory_usage(proc=(
            self.function,
            (train_loader, epoch, device, optimizer, scheduler)),
            max_usage=True, retval=True)
        # self.function(*args, **kwargs)
        self.pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        self.stream.write(s.getvalue())
        self.stream.write('\n\n')
        self.stream.write(f"GPU: {self.gpu.mem_used}")
        self.stream.write('\n\n')
        self.stream.write(f"MEM: {result[0]}")
        return result[1]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stream.close()
