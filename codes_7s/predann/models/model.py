import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def initialize(self, m):
        if isinstance(m, (nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
