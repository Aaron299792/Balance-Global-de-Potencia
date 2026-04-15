import torch
from scipy.constants import e
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("../"))

from modules import sigmav

filepath = "../data/" + sys.argv[1]

sigma = sigmav(filepath)

T_min = 10**sigma.lTadf11_t[0]
T_max = 10**sigma.lTadf11_t[-1]

T_test = torch.linspace(T_min * 1.01, T_max * 0.99, 2000, device='cpu', dtype=torch.float64, requires_grad=True)

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

plt.title(r'Adimentional $\langle \sigma v \rangle$ rates')
plt.xlabel(r'Normalized temperature $\hat{T}$')
plt.ylabel(r'$\frac{\langle \sigma v \rangle(T_{max}\hat{T})}{\langle \sigma v \rangle(T_{max})}$')
plt.scatter(sigma.lTadf11_t.detach().cpu().numpy(), (sigma.lradf11_1D_t - sigma.lradf11_1D_t[-1].detach()).detach().cpu().numpy(), label=f'adf11 for {filepath}')
plt.plot(torch.log10(T_test).detach().cpu().numpy(), torch.log10(R_scaled).detach().cpu().numpy(), label="linear interpolation for test tensor")
plt.legend()
plt.show()





