import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

device = torch.device("cuda" if torch.device.is_available() else "cpu")

logabs = lambda x: torch.log(torch.abs(x))


class Actnorm(nn.Module):
    def __init__(self, in_chall, logdet=True):
        super(Actnorm, self).__init__()

        self.loc = nn.parameter(torch.zeros(1, in_chall, 1, 1))
        self.scale = nn.parameter(torch.ones(1, in_chall, 1, 1))

        self.initialized = nn.parameter(torch.tensor(0, dtype=torch.uint8), requires=False)
        self.logdet = logdet
    def initialize(self, input):
        with torch.no_grad:
            mean = torch.mean(input, dim=(0, 2, 3), keepdim=True)
            std = torch.std(input, dim=(0, 2, 3), keepdim=True)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1/(std + 1e-6))
    def forward(self, input):
        batch_size, _, height, width = input.shape
        if self.initialized.item == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        logdet = torch.tile(torch.tensor([logdet], device=device), (batch_size, ))

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)
    def reverse(self, output):
        return output / self.scale - self.loc

class InConv2d(nn.Module):
    def __init__(self, in_channel):
        super(InConv2d, self).__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.parameter(weight)

    def forward(self, input):
        batch_size, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeeze().double()[1].float())
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeeze().inverse().unsqueeze(2).unsqueeze(3))




