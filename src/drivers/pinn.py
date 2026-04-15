# -------------------------------
# Modules
# -------------------------------

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, e, electron_mass
from cherab.core.atomic import hydrogen
from cherab.openadas.parse.adf11 import parse_adf11

# -------------------------------
# Global Constants and Parameters
# -------------------------------
EPS = 1e-35
DTYPE = torch.float64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


