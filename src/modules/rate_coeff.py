# -------------------------------
# Modules
# -------------------------------
import numpy as np
import torch
from cherab.core.atomic import hydrogen
from cherab.openadas.parse.adf11 import parse_adf11

import os
import sys

sys.path.append(os.path.abspath('../../'))

# -------------------------------
# Class
# -------------------------------
class sigmav:
    def __init__(self, filepath, dtype=torch.float64, device='cpu', epsilon=1e-40):
        self.data = parse_adf11(hydrogen, filepath)
        self.dtype = dtype
        self.device = device
        self.epsilon = epsilon
        self.lTadf11_t = self._to_tensor(self._extract_data('te'))
        self.lradf11_1D_t = self._to_tensor(self._extract_data('rates')).mean(dim=0)
        self.rate_ref = 10 ** self.lradf11_1D_t.mean()

    def _extract_data(self, value):
        key = list(self.data.keys())[0]
        return self.data[key][1][value]

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)


    def _interp_lrate(self, ly):
        ly = ly.clamp(self.lTadf11_t[0] + 1e-12, self.lTadf11_t[-1] - 1e-12)
        idx = (torch.searchsorted(self.lTadf11_t, ly) - 1).clamp(0, self.lTadf11_t.size(0))

        x0 = self.lTadf11_t[idx]
        x1 = self.lTadf11_t[idx + 1]
        w = (ly - x0) / (x1 - x0)
        y0 = self.lradf11_1D_t[idx]
        y1 = self.lradf11_1D_t[idx + 1]

        return (1 - w) * y0 + w * y1

    def rates(self, T):
        T = T.clamp(min=self.epsilon)
        lT = torch.log10(T)
        lrate = self._interp_lrate(lT)

        return 10 ** lrate / self.rate_ref

