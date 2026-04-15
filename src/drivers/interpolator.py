import numpy as np
import matplotlib.pyplot as plt
from cherab.openadas.parse.adf11 import parse_adf11
from cherab.core.atomic import hydrogen
from scipy.interpolate import RegularGridInterpolator

import sys
import os

sys.path.append(os.path.abspath('../'))

data = parse_adf11(hydrogen, '../data/scd96_h.dat')
key = list(data.keys())[0]

lden =  np.array(data[key][1]['ne'])
ltemp = np.array(data[key][1]['te'])
lrates = np.array(data[key][1]['rates'])

interp = RegularGridInterpolator((lden, ltemp), lrates)

ln = np.linspace(lden.min(), lden.max(), 80)
lt = np.linspace(ltemp.min(), ltemp.max(),100)

X, Y = np.meshgrid(ln, lt, indexing='ij')

points = np.column_stack((X.ravel(), Y.ravel()))

lr = interp(points).reshape(80,100)

plt.scatter(ltemp, lrates.mean(axis=0))
plt.plot(lt, lr.mean(axis=0))
plt.show()
