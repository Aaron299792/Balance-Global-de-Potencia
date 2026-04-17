import numpy as np
import torch
from scipy.constants import e
import matplotlib.pyplot as plt
from cherab.core.atomic import hydrogen
import sys
import os

sys.path.append(os.path.abspath("../"))

from modules import sigmav

filepath = "../../data/" + sys.argv[1]

sigma = sigmav(filepath, hydrogen)

lT_min = sigma.lTadf11[0]
lT_max = sigma.lTadf11[-1]

T_test_n = np.power(np.linspace(2.5e-5, 1, 1000), 10)
T_test = torch.tensor(T_test_n, dtype = torch.float64, device='cpu', requires_grad=True)

R_scaled = sigma.rates(T_test)

loss = R_scaled.mean()
loss.backward()

print("R_scaled mean:", R_scaled.mean().item())
print("R_scaled std :", R_scaled.std().item())
print("Grad mean:", T_test.grad.abs().mean().item())
print("Grad max :", T_test.grad.abs().max().item())
print("Finite R:", torch.isfinite(R_scaled).all().item())
print("Finite grad:", torch.isfinite(T_test.grad).all().item())
print("logT range:", torch.log10(T_test).min().item(), torch.log10(T_test).max().item())

"""
plt.title(r'Adimentional $\langle \sigma v \rangle$ rates')
plt.xlabel(r'Normalized temperature $\hat{T}$')
plt.ylabel(r'$\frac{\langle \sigma v \rangle(T_{max}\hat{T})}{\langle \sigma v \rangle(T_{max})}$')
plt.plot( (sigma.lTadf11 - sigma.lTadf11[-1]).detach().cpu().numpy(), (sigma.lrates_uniform - sigma.lrates_uniform[-1]).detach().cpu().numpy(), label=f'log rates for file={sys.argv[1]}')
plt.plot(torch.log10(T_test).detach().cpu().numpy(), torch.log10(R_scaled).detach().cpu().numpy(), '-.',label="linear interpolation for test tensor")
plt.xlim((lT_min - lT_max).detach().cpu(), 0)
plt.ylim(sigma.lrates_uniform[0] - sigma.lrates_uniform[-1], 1.0)
plt.legend()
plt.show()
"""
