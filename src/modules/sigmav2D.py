import torch
from cherab.core.atomic import hydrogen
from cherab.openadas.parse.adf11 import parse_adf11
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class RateCoeff2D:
    def __init__(self, file, gas, n_max, T_max, dtype=torch.float64, device="cpu", Nn=200, Nt=200):
        self.dtype = dtype
        self.device = device

        self.data = parse_adf11(gas, file)

        lden = self._extract_data('ne')
        ltemp = self._extract_data('te')
        lrates = self._extract_data('rates')

        ln = np.linspace(lden.min(), lden.max(), Nn)
        lt = np.linspace(ltemp.min(), ltemp.max(), Nt)

        interp = RegularGridInterpolator((lden, ltemp), lrates)

        X, Y = np.meshgrid(ln, lt, indexing='ij')
        pts = np.column_stack((X.ravel(), Y.ravel()))
        lr = interp(pts).reshape(Nn, Nt)

        self.ln = torch.as_tensor(ln, dtype=torch.float64, device=self.device)
        self.lt = torch.as_tensor(lt, dtype=torch.float64, device=self.device)
        self.lr = torch.as_tensor(lr, dtype=torch.float64, device=self.device)
        self.n_max = n_max
        self.T_max = T_max

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

    def _bilinear(self, lnq, ltq):
        shape = lnq.shape

        x = lnq.reshape(-1)
        y = ltq.reshape(-1)

        ix = (torch.searchsorted(self.ln, x) - 1).clamp(0, self.ln.size(0) - 2)
        iy = (torch.searchsorted(self.lt, y) - 1).clamp(0, self.lt.size(0) - 2)

        x0 = self.ln[ix]
        x1 = self.ln[ix + 1]
        y0 = self.lt[iy]
        y1 = self.lt[iy + 1]

        f00 = self.lr[ix, iy]
        f01 = self.lr[ix, iy + 1]
        f10 = self.lr[ix + 1, iy]
        f11 = self.lr[ix + 1, iy + 1]

        wx = (x - x0) / (x1 - x0 + 1e-12)
        wy = (y - y0) / (y1 - y0 + 1e-12)

        out = (
            (1 - wx) * (1 - wy) * f00 +
            (1 - wx) * wy       * f01 +
            wx       * (1 - wy) * f10 +
            wx       * wy       * f11
        )

        return out.reshape(shape)

    def rate(self, n_norm, T_norm): #De momento no trabaja para T y n normalizado, tengo que corregir
        ne = n_norm * self.n_max
        Te = T_norm * self.T_max
        lnq = torch.log10(ne)
        ltq = torch.log10(Te)

        lr = self._bilinear(lnq, ltq)

        return 10.0 ** lr
