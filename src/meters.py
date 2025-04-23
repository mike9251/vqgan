import torch
import torch.distributed as dist
from collections import deque


class RunningMeter:
    def __init__(self, window_size: int, ddp: bool = False):
        self.queue = deque(maxlen=window_size)
        self.ddp = ddp
    
    def update(self, value: float):
        self.queue.append(value)

    def __call__(self, value: float):
        self.update(value)
    
    def compute(self):
        avg_value = sum(self.queue) / len(self.queue)
        if self.ddp:
            avg_value = torch.tensor([avg_value], requires_grad=False).cuda()
            dist.all_reduce(avg_value, op=dist.ReduceOp.AVG)
            return avg_value.cpu().item()

        return avg_value

