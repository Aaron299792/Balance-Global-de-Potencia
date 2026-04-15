import torch
from scipy.constants import e
import matplotlib.pyplot as plt
from cherab.core.atomic import hydrogen
import sys
import os

sys.path.append(os.path.abspath("../"))

from modules import sigmav

filepath = "../data/" + sys.argv[1]

sigma = sigmav(filepath, hydrogen)

lT_min = sigma.lTadf11[0]
lT_max = sigma.lTadf11[-1]

T_test = torch.pow(torch.linspace(lT_min, lT_max, 100), 10)

R_scaled = sigma.rates(T_test)

#loss = R_scaled.mean()
#loss.backward()

#print("R_scaled mean:", R_scaled.mean().item())
#print("R_scaled std :", R_scaled.std().item())
#print("Grad mean:", T_test.grad.abs().mean().item())
#print("Grad max :", T_test.grad.abs().max().item())
#print("Finite R:", torch.isfinite(R_scaled).all().item())
#print("Finite grad:", torch.isfinite(T_test.grad).all().item())
#print("logT range:", torch.log10(T_test).min().item(), torch.log10(T_test).max().item())

print(T_test)
print(R_scaled)
plt.title(r'Adimentional $\langle \sigma v \rangle$ rates')
plt.xlabel(r'Normalized temperature $\hat{T}$')
plt.ylabel(r'$\frac{\langle \sigma v \rangle(T_{max}\hat{T})}{\langle \sigma v \rangle(T_{max})}$')
plt.plot(sigma.lTadf11.detach().cpu().numpy(), sigma.lrates_uniform.detach().cpu().numpy(), label=f'log rates for file={sys.argv[1]}')
plt.plot(torch.log10(T_test).detach().cpu().numpy(), torch.log10(R_scaled).detach().cpu().numpy(), label="linear interpolation for test tensor")
plt.xlim(lT_min.detach().cpu(), lT_max.detach().cpu())
plt.ylim(sigma.lrates_uniform[0].detach().cpu(), sigma.lrates_uniform[-1].detach().cpu() + 1)
plt.legend()
plt.show()





