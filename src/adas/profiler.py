from typing import Tuple, List
from pstats import SortKey
from pathlib import Path

import cProfile
import pstats
import io

from memory_profiler import memory_usage

from .components import Statistics
from .utils import pstats_to_dict
from .gpu import GPU


class Profiler:
    gpu_id: int = 0
    root: Path = Path('.')

    def __init__(self, function):
        self.gpu = GPU(Profiler.gpu_id)
        self.stream = (Profiler.root / 'stats.csv').open('w+')
        self.pr = cProfile.Profile()
        self.function = function
        self.statistics = List[Statistics]

    def __call__(self, train_loader, epoch: int,
                 device, optimizer, scheduler) -> Tuple[float, float]:
        self.gpu.update()
        self.pr.enable()
        result = memory_usage(proc=(
            self.function,
            (train_loader, epoch, device, optimizer, scheduler)),
            max_usage=True, retval=True)
        self.pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats("epoch_iteration|step|trial_iteration|test_main")
        stats_list = pstats_to_dict(s.getvalue())
        header = 'epoch,epoch_gpu_mem_used,epoch_ram_used'
        content = f'{epoch},{self.gpu.mem_used},{result[0]}'
        for stat in stats_list:
            header += f",{stat['name']}_n_calls"
            header += f",{stat['name']}_tot_time"
            header += f",{stat['name']}_per_call1"
            header += f",{stat['name']}_cum_time"
            header += f",{stat['name']}_per_call2"
            content += f",{stat['n_calls']}"
            content += f",{stat['tot_time']}"
            content += f",{stat['per_call1']}"
            content += f",{stat['cum_time']}"
            content += f",{stat['per_call2']}"
        header += '\n'
        content += '\n'
        self.stream.write(header)
        self.stream.write(content)
        return result[1]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stream.close()
