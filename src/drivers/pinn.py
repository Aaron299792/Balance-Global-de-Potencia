# -------------------------------
# Modules
# -------------------------------
import numpy as np
import torch
import torch.nn.functional as F
import deepxde as dde
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, e, electron_mass
from cherab.core.atomic import hydrogen
from cherab.openadas.parse.adf11 import parse_adf11

import sys
import os

sys.path.append(os.path.abspath('../modules'))

from sigmav2D import RateCoeff2D

# -------------------------------
# Global Constants and Parameters
# -------------------------------
B = 1 #T
E_ION = 13.6 #eV
T_MAX = 7000 # eV
N_MAX = 2e13 # cm^-3
R = 72 # cm
D_0 = 1e4 * (2 * np.sqrt(2 * np.pi * electron_mass) / 3) * (e / (4 * np.pi * epsilon_0 * B))**2 * (N_MAX * 1e6  / np.sqrt(T_MAX * e)) #cm^2 s^-1
KAPPA_0 = 4.7 * N_MAX * D_0 # cm^-1 s^-1
C = 2 * np.log(4 * np.pi * np.power(epsilon_0 * T_MAX * e, 1.5) / (e**3 * np.sqrt(N_MAX)))
LAMBDA_N = 1.0
LAMBDA_T = 1.0
P_EXT = 1e18 #eV cm^-3 s^-1
VAR = 0.15
MU = 0.1
N_0 = 1e8 # cm^{-3}

# -------------------------------
# Device Settings
# -------------------------------
DTYPE = torch.float64
EPS   = 1e-40
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(DEVICE)


"""------------------------------
Rate objects

Particle sources and drains:
    sigma_ion: <sigma nu>_ion n0 * n
    sigma_rec: <sigma nu>_rec n^2

power loss rates:
    p_rec = <E_rec><sigma nu>_rec n ^ 2 (contains Bremsstrahlung radiation)
    p_rad_i: <E_rad^i><sigma nu>_rad^i n^2
    p_rad_0: <E_rad^0><sigma nu>_rad^0 n
------------------------------"""

sigma_ion = RateCoeff2D('../../data/scd96_h.dat', hydrogen, n_max=N_MAX, T_max = T_MAX)
sigma_rec = RateCoeff2D('../../data/acd96_h.dat', hydrogen, n_max=N_MAX, T_max = T_MAX)
p_rec = RateCoeff2D('../../data/prb96_h.dat', hydrogen, n_max=N_MAX, T_max = T_MAX)
p_rad_i = RateCoeff2D('../../data/plt96_h.dat', hydrogen, n_max=N_MAX, T_max = T_MAX)
p_rad_0 = RateCoeff2D('../../data/prc96_h.dat', hydrogen, n_max=N_MAX, T_max = T_MAX)


# -------------------------------
# Functions
# -------------------------------

def ode_system(X, y):
    #----------------------------
    # Preamble
    #----------------------------
    rho = X[:, 0:1]
    rho_eps = torch.clamp(rho, min=1e-6) #no puede ser cero, las ecs explotan
    n_hat = y[:, 0:1]
    T_hat = y[:, 1:2]

    n = n_hat * N_MAX
    T = T_hat * T_MAX

    dn_rho = dde.grad.jacobian(y, X, i=0, j=0) #es sobre las normalizadas
    dT_rho = dde.grad.jacobian(y, X, i=1, j=0)

    # T^3/n and log(T^3/n)
    T3n = torch.clamp((T_hat ** 3) / (n_hat + EPS), min=1e-12,max=1e12)
    logT3n = torch.log(T3n)
    D_hat = n_hat / torch.sqrt(T_hat + EPS) * (logT3n + C)
    D_hat = torch.clamp(D_hat, min=1e-8, max=1e8)

    #----------------------------
    # ODE: Particles
    #----------------------------
    ionization = torch.nan_to_num(sigma_ion.rate(n, T), nan=0.0)
    recombination = torch.nan_to_num(sigma_rec.rate(n, T), nan=0.0)


    S_particle = (R**2 / (D_0 * N_MAX)) * (ionization - recombination)
    flux_n = rho_eps * D_hat * dn_rho
    dflux_n_rho = dde.grad.jacobian(flux_n, X, i=0, j=0)

    ode1 = dflux_n_rho / rho_eps  + S_particle

    #----------------------------
    # ODE: Energy
    #----------------------------

    flux_T = rho_eps * D_hat * dT_rho
    dflux_T_rho = dde.grad.jacobian(flux_T, X, i=0, j=0)
    conduction = dflux_T_rho / rho_eps

    gaussian_profile = torch.exp(-0.5 * ((rho - MU) / VAR)**2)
    power_dep = (P_EXT * R / (N_MAX * T_MAX * D_0)) * gaussian_profile

    P_rec = torch.nan_to_num(p_rec.rate(n, T), nan=0.0)
    P_rad_i = torch.nan_to_num(p_rad_i.rate(n,T), nan=0.0)
    P_rad_0 = torch.nan_to_num(p_rad_0.rate(n,T), nan=0.0)

    P_loss = P_rec + P_rad_i + P_rad_0

    loss_power = (R**2 / (N_MAX * T_MAX * D_0)) * P_loss

    ode2 = conduction + power_dep - loss_power #me falta agregar coulomb

    ode1 = torch.nan_to_num(ode1, nan=0.0)
    ode2 = torch.nan_to_num(ode2, nan=0.0)

    return [ode1, ode2]

def boundary_l(X, on_boundary):
    return on_boundary and dde.utils.isclose(X[0], 0)

def boundary_r(X, on_boundary):
    return on_boundary and dde.utils.isclose(X[0], 1)

def robin_n(X, y):
    n_hat = y[:, 0:1]
    T_hat = y[:, 1:2]

    # T^3/n and log(T^3/n)
    T3n = torch.clamp((T_hat ** 3) / (n_hat + EPS), min=1e-12,max=1e12)
    logT3n = torch.log(T3n)
    D_hat = n_hat / torch.sqrt(T_hat + EPS) * (logT3n + C)
    D_hat = torch.clamp(D_hat, min=1e-8, max=1e8)

    return -n_hat / (LAMBDA_N * D_hat)

def robin_T(X, y):
    n_hat = y[:, 0:1]
    T_hat = y[:, 1:2]

    # T^3/n and log(T^3/n)
    T3n = torch.clamp((T_hat ** 3) / (n_hat + EPS), min=1e-12,max=1e12)
    logT3n = torch.log(T3n)
    D_hat = n_hat / torch.sqrt(T_hat + EPS) * (logT3n + C)
    D_hat = torch.clamp(D_hat, min=1e-8, max=1e8)

    return -T_hat / (LAMBDA_T * D_hat)


#--------------------------------
# Main
# -------------------------------
geom = dde.geometry.Interval(0.0, 1.0)

bc_l_n = dde.icbc.DirichletBC(geom, lambda x: 1.0, boundary_l, component=0)
bc_l_T = dde.icbc.DirichletBC(geom, lambda x: 1.0, boundary_l, component=1)

bc_r_n = dde.icbc.RobinBC(geom, robin_n, boundary_r, component=0)
bc_r_T = dde.icbc.RobinBC(geom, robin_T, boundary_r, component=1)

boundaries = [bc_l_n, bc_l_T, bc_r_n, bc_r_T]

data = dde.data.PDE(geom, ode_system, boundaries, 300, 2, num_test=300)


net = dde.nn.FNN([1] + [64]*6 + [2], "tanh", "Glorot normal")

def output_transform(x, y):
    return torch.exp(y)

net.apply_output_transform(output_transform)

net.apply_output_transform(lambda x, y: torch.exp(y))

model = dde.Model(data, net)

loss_weights = [100, 100, 1, 1, 100, 100]

model.compile("adam", lr=1e-3, loss_weights=loss_weights)
model.train(iterations=5000)

model.compile("adam", lr=1e-4, loss_weights=loss_weights)
model.train(iterations=10000)

model.compile("L-BFGS")
model.train()

x = np.linspace(0, 1, 500)[:, None]
y = model.predict(x)

n_pred = y[:, 0]
T_pred = y[:, 1]

plt.figure()
plt.plot(x, n_pred)
plt.xlabel("x")
plt.ylabel("n(x)")
plt.title("Density")
plt.grid()
plt.savefig('../../plots/density_profile.pdf', format='pdf', bbox_inches='tight')

plt.figure()
plt.plot(x, T_pred)
plt.xlabel("x")
plt.ylabel("T(x)")
plt.title("Temperature")
plt.grid()
plt.savefig('../../plots/temperature_profile.pdf', format='pdf', bbox_inches='tight')
