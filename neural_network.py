"""
Definition of the neural network to be used

@author: eduardokapp
"""

import torch
import torch.nn as nn
import numpy as np

class network(nn.Module):
    def __init__(self, input_size, output_size):
        super(network, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, output_size)

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.l2(out1)
        return out2
