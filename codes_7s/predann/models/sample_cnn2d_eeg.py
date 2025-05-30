import torch
import torch.nn as nn
from .model import Model
import numpy as np
from simclr.modules.identity import Identity
import torch.nn.functional as F
from torch.autograd import Function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SampleCNN2DEEG(Model):
    def __init__(self, out_dim, kernal_size):
        super(SampleCNN2DEEG, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 32, kernal_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernal_size, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernal_size, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.5)    
        self.fc = Identity()
        self.projector1 = nn.Sequential(
            nn.Linear(128, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, out_dim, bias=False),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(128, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100, bias=False),
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.sequential(x)
        out = self.dropout(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        logit2 = self.projector2(out)
        logit1 = self.projector1(out)
        return logit1, logit2
