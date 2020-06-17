from .gpu import GPU


class Profiler:
    def __init__(self, gpu_id: int = 0):
        self.gpu = GPU(gpu_id)
