
import abc
import torch
import torch.nn as nn
import numpy as np


class Noise(abc.ABC, nn.Module):
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)
    @abc.abstractmethod
    def rate_noise(self, t):
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        pass


class LogLinearNoise(Noise, nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)
