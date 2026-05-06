# -------------------------------
# Modules
# -------------------------------
import numpy as np
import torch

import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from cherab.core.atomic import hydrogen
from cherab.openadas.parse.adf11 import parse_adf11

# -------------------------------
# Class
# -------------------------------

class sigmav:
    """
    Object that works with tensors. It takes a file adf11 from OpenADAS and transforms it into a regular 1D grid of high resolution. This
    can be interpolate as a tensor to give the collision rates < \sigma v > from ADAS data base. One could get the temperature data from the file as well.
    """
    def __init__(self, filepath, gas, T_max = None, dtype=torch.float64, device='cpu', epsilon=1e-40):
        self.data = parse_adf11(gas, filepath)
        self.dtype = dtype
        self.device = device
        self.epsilon = epsilon
        self.lrates_uniform, self.lTadf11 = self._uniform_interpolation()
        self.ltmax = self.lTadf11[-1] if T_max == None else np.log(T_max)

    def _extract_data(self, value):
        """
        helper internal function.It finds and extracts the data from the recursive dictionary return by parse_adf11

        Arguments:
        -------------------------------------------------------------------------
            value: str,
                could be 'ne': density, 'te': temperature, 'rates': rates.

        Return:
            out: n_array,
                returns an array that contains the value extracted from the adf11 file.
        """
        key = list(self.data.keys())[0]
        return np.array(self.data[key][1][value], dtype=np.float64)

    def _uniform_interpolation(self):
        """
        helper internal function. RegularGridInterpolation. Uses the data from an adf11 file and creates a 1D uniform grid to initialize lrates_uniform atribute.
        Computes and initializes the rates for a 2D grid and then averages over the density axis by extracting the data then creating a rectangular grid.
        Arguments: None, uses data atribute, and _extract_data function.

        Return:
        -------------------------------------------------------------------------
            out: Tensor,
                1D tensor with a high resolution uniform data.
        """
        lden = self._extract_data('ne')
        ltemp = self._extract_data('te')
        lrates = self._extract_data('rates')

        interp = RegularGridInterpolator((lden, ltemp), lrates)
        ln = np.linspace(lden.min(), lden.max(), 5000)
        lt = np.linspace(ltemp.min(), ltemp.max(),5000)
        X, Y = np.meshgrid(ln, lt, indexing='ij')
        points = np.column_stack((X.ravel(), Y.ravel()))
        lr = interp(points).reshape(len(ln),len(lt)).mean(axis=0)

        return torch.as_tensor(lr, dtype=self.dtype, device=self.device), torch.as_tensor(lt, dtype = self.dtype, device=self.device)

    def _interp_lrate(self, ly):
        """
        helper function. Computes a linear interpolation for a tensor. It preserves the tensor graphs and memory of operations, such that AD is not broken within the process. Given
        a point x_{i} it assumes the next point x_{i+1} is such that x_{i+1} = w*x_{i} + b.

        Arguments:
        --------------------------------------------------------------------------
        ly: Tensor,
            1D tensor of temperatures with data to be allocated within a value of the rates.

        Return:
            out: Tensor,
                1D tensor of the rates evaluated at the temperaturs in the tensor ly.
        """
        shape = ly.shape
        ly_flat = ly.reshape(-1)

        idx = (torch.searchsorted(self.lTadf11, ly_flat) - 1).clamp(0, self.lTadf11.size(0) - 2)
        x0 = self.lTadf11[idx]
        x1 = self.lTadf11[idx + 1]
        w = (ly_flat - x0) / (x1 - x0)
        y0 = self.lrates_uniform[idx]
        y1 = self.lrates_uniform[idx + 1]
        out = (1 - w) * y0 + w * y1
        return out.reshape(shape)

    def rates(self, T_normalized):
        """
        method that takes a normalized temperature and returns the normalized process rates.

        Arguments:
        ---------------------------------------------------------------------------
            T_normalized: Tensor,
                1D tensor normalized temperature in a interval ]0, 1].

        Return:
        ---------------------------------------------------------------------------
            out: Tensor,
                1D tensor with preserve AD, contains the rates of the process at a temperature T divided by the rates evaluated at T_max.
        """
        T = T_normalized *  10**self.ltmax
        T = T.clamp(min=self.epsilon)
        lT = torch.log10(T)
        lrate = self._interp_lrate(lT)
        lrate_ref = self._interp_lrate(self.ltmax)
        return 10 ** lrate
