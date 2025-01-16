import torch.nn
import torch.optim as optim


class Optimizer:
    def __init__(self, cfgs, model: torch.nn.Module):
        self.cfgs = cfgs
        self.model = model
        self.param_groups = [{
            'params': self.model.parameters(),
            'lr': self.cfgs['optimizer']['lr']
        }]

    def get_optim(self):
        if self.cfgs["optimizer"]["type"].lower() == "sgd":
            optimizer = optim.SGD(self.param_groups, momentum=self.cfgs["OPTIMIZER"]["MOMENTUM"])

        elif self.cfgs['optimizer']['type'].lower() == 'adamw':
            optimizer = optim.AdamW(self.param_groups)

        return optimizer
