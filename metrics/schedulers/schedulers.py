import math
import torch
import torch.optim.lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


class Scheduler():
    def __init__(self, cfgs, optimizer: Optimizer):
        self.cfgs = cfgs
        self.optimizer = optimizer

    def get_scheduler(self) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        if self.cfgs['lr_scheduler']['type'] == 'cosine_lr':
            scheduler = CosineAnnealingLR(self.optimizer, self.cfgs['num_epochs'])
        if self.cfgs['lr_scheduler']['type'] == 'polynomial_lr':
            scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=1000, power=0.2)
        return scheduler
