from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch

class TensorboardLogger:
    def __init__(self, log_dir: str = "logs"):
        self.writer = SummaryWriter(log_dir)
    
    def log(self, logs: dict, global_step: int):
        for key, value in logs.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar(key, value, global_step)
            elif isinstance(value, torch.Tensor) and len(value.shape) > 3:
                grid = torchvision.utils.make_grid(value)
                self.writer.add_image(key, grid, global_step)

