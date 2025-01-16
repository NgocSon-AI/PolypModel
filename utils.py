import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import argparse
import yaml
import wandb
import random
import csv
import glob
import json
import shutil
import traceback
import gc
import copy
import time
import pickle
import numpy as np


from datetime import datetime
from collections import OrderedDict

from data.dataloader import TrainDataset, EvalDataset
from data.dataloader import get_loader

from models.model import Meta_Polypv2
from metrics.schedulers.schedulers import Scheduler
from metrics.optimizers.optimizers import Optimizer
from metrics.losses.losses import MetaPolypv2_Loss
from metrics.metrics import Dice_Coeff, IoU


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)