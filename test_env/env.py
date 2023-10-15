import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from pathlib import Path


class Environment:
    datapath = Path('../data/')

    def __init__(self, model, learner, runner,
                 dataset: str = 'mnist',
                 device: str = 'gpu'
                 ):
        self.model = model
        self.learner = learner
        self.runner = runner
        self.train_data = None
        self.test_data = None

        match dataset:
            case 'mnist':
                self.train_data = torchvision.datasets.MNIST(
                    root=str(Environment.datapath), train=True, download=True, transform=None)
                self.test_data = torchvision.datasets.MNIST(
                    root=str(Environment.datapath), train=False, download=True, transform=None)

