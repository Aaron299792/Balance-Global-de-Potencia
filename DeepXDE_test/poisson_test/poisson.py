import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

dde.config.set_default_float("float32")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(device)

# ── Physics ───────────────────────────────────────────────────────────────────
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - 2


# ── Domain ────────────────────────────────────────────────────────────────────
geom = dde.geometry.Interval(-1, 1)

# Dirichlet BC: u(-1) = 0
bc_dirichlet = dde.icbc.DirichletBC(
    geom,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], -1)
)

# Neumann BC: u'(1) = -2  (from d/dx[1 - x²] = -2x → at x=1: -2)
bc_neumann = dde.icbc.NeumannBC(
    geom,
    lambda x: -2,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1)
)

data = dde.data.PDE(
    geom,
    pde,
    [bc_dirichlet, bc_neumann],  # both BCs, still no solution
    num_domain=64,
    num_boundary=4,
    num_test=200,
)

# ── Network ───────────────────────────────────────────────────────────────────
net = dde.nn.FNN([1] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.net.to(device)

# ── Phase 1: Adam ─────────────────────────────────────────────────────────────
print("=" * 55)
print("Phase 1: Adam optimizer")
print("=" * 55)
model.compile("adam", lr=1e-3)
losshistory_adam, train_state_adam = model.train(iterations=10000, display_every=1000)

# ── Phase 2: L-BFGS refinement ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("Phase 2: L-BFGS refinement")
print("=" * 55)
model.compile("L-BFGS")
losshistory_lbfgs, train_state_lbfgs = model.train(display_every=200)

# ── Compare with exact solution (only NOW) ────────────────────────────────────
print("\n" + "=" * 55)
print("Comparing model output with exact solution...")
print("=" * 55)

def exact(x):
    return 1 - x ** 2

def exact_deriv(x):
    return -2 * x

x_test = np.linspace(-1, 1, 500).reshape(-1, 1).astype(np.float32)
x_tensor = torch.tensor(x_test).to(device)
x_tensor.requires_grad_(True)

model.net.eval()
y_pred_tensor = model.net(x_tensor)

# Compute predicted derivative via autograd
dy_pred_tensor = torch.autograd.grad(
    y_pred_tensor, x_tensor,
    grad_outputs=torch.ones_like(y_pred_tensor),
    create_graph=False
)[0]

y_pred = y_pred_tensor.detach().cpu().numpy()
dy_pred = dy_pred_tensor.detach().cpu().numpy()
y_exact = exact(x_test)
dy_exact = exact_deriv(x_test)

l2_error = np.linalg.norm(y_pred - y_exact) / np.linalg.norm(y_exact)
max_error = np.max(np.abs(y_pred - y_exact))
neumann_residual = np.abs(dy_pred[-1] - (-2))  # check u'(1) ≈ -2

print(f"  L2 relative error      : {l2_error:.6f}")
print(f"  Max absolute error      : {max_error:.6f}")
print(f"  Neumann residual at x=1 : {neumann_residual[0]:.6f}  (u'(1) predicted: {dy_pred[-1][0]:.4f}, expected: -2.0000)")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1. Solution comparison
axes[0, 0].plot(x_test, y_exact, "k-", linewidth=2, label="Exact: 1 − x²")
axes[0, 0].plot(x_test, y_pred, "r--", linewidth=2, label="PINN prediction")
axes[0, 0].set_title("Solution Comparison")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("u(x)")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Derivative comparison
axes[0, 1].plot(x_test, dy_exact, "k-", linewidth=2, label="Exact: −2x")
axes[0, 1].plot(x_test, dy_pred, "r--", linewidth=2, label="PINN du/dx")
axes[0, 1].axvline(x=1, color="blue", linestyle=":", linewidth=1.5, label="Neumann BC at x=1")
axes[0, 1].axvline(x=-1, color="green", linestyle=":", linewidth=1.5, label="Dirichlet BC at x=-1")
axes[0, 1].set_title("First Derivative Comparison")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("u'(x)")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Pointwise error
axes[1, 0].plot(x_test, np.abs(y_pred - y_exact), "b-", linewidth=1.5, label="|u error|")
axes[1, 0].plot(x_test, np.abs(dy_pred - dy_exact), "orange", linewidth=1.5, label="|u' error|")
axes[1, 0].set_title("Pointwise Absolute Error")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("Error")
axes[1, 0].set_yscale("log")
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Training loss (Adam + L-BFGS)
adam_steps = len(losshistory_adam.loss_train)
lbfgs_steps = len(losshistory_lbfgs.loss_train)
adam_total = [sum(l) for l in losshistory_adam.loss_train]
lbfgs_total = [sum(l) for l in losshistory_lbfgs.loss_train]

axes[1, 1].semilogy(range(adam_steps), adam_total, "b-", label="Adam", linewidth=1.5)
axes[1, 1].semilogy(range(adam_steps, adam_steps + lbfgs_steps), lbfgs_total, "r-", label="L-BFGS", linewidth=1.5)
axes[1, 1].axvline(x=adam_steps, color="gray", linestyle="--", label="Switch point")
axes[1, 1].set_title("Training Loss")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("Total loss")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.suptitle(f"1D Poisson PINN — Dirichlet + Neumann BCs\nL2 error: {l2_error:.2e} | Neumann residual: {neumann_residual[0]:.2e}", fontsize=12)
plt.tight_layout()
plt.savefig("poisson_neumann.png", dpi=150)
plt.show()
print("\nPlot saved to poisson_neumann.png")
