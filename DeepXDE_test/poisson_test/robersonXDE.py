import os
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")
print(f"Using device: {device}")

# ── Constants ─────────────────────────────────────────────────────────────────
K1   = 0.04
K2   = 1e4
K3   = 3e7
T_END = 100.0

# ── Reference solution ────────────────────────────────────────────────────────
def robertson_ode(t, y):
    return [
        -K1*y[0] + K2*y[1]*y[2],
         K1*y[0] - K2*y[1]*y[2] - K3*y[1]**2,
         K3*y[1]**2
    ]

print("Computing reference solution...")
ref = solve_ivp(robertson_ode, [0, T_END], [1, 0, 0],
                method="Radau", dense_output=True, rtol=1e-12, atol=1e-14)
print(f"  Reference solved: {ref.message}")


# ── Network with residual connections ─────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l1  = nn.Linear(width, width)
        self.l2  = nn.Linear(width, width)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.l2(self.act(self.l1(x)))) + x


class RobertsonNet(nn.Module):
    """
    Outputs (y1, log10(y2), y3) internally, but forward() returns (y1, y2, y3).
    Log-transforming y2 is critical: it spans ~8 orders of magnitude.
    y1 and y3 are kept raw but passed through softplus to stay non-negative.
    """
    def __init__(self, width=128, n_blocks=6):
        super().__init__()
        self.input_layer  = nn.Linear(1, width)
        self.blocks       = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(width, 3)
        self.act          = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward_raw(self, t):
        """Returns raw network output: (y1_raw, log10y2_raw, y3_raw)"""
        x = self.act(self.input_layer(t))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

    def forward(self, t):
        raw = self.forward_raw(t)
        y1       = torch.nn.functional.softplus(raw[:, 0:1])
        log10_y2 = raw[:, 1:2]
        y2       = 10 ** log10_y2
        y3       = torch.nn.functional.softplus(raw[:, 2:3])
        return torch.cat([y1, y2, y3], dim=1)


# ── Loss computation ──────────────────────────────────────────────────────────
def compute_residuals(net, t):
    """Returns PDE residuals with full autograd graph."""
    t = t.requires_grad_(True)
    y = net(t)
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    def dy(yi):
        return torch.autograd.grad(
            yi, t,
            grad_outputs=torch.ones_like(yi),
            create_graph=True, retain_graph=True
        )[0]

    dy1 = dy(y1);  dy2 = dy(y2);  dy3 = dy(y3)

    r1 = dy1 + K1*y1 - K2*y2*y3
    r2 = dy2 - K1*y1 + K2*y2*y3 + K3*y2**2
    r3 = dy3 - K3*y2**2

    # Conservation: y1 + y2 + y3 should equal 1
    conservation = y1 + y2 + y3 - 1.0

    return r1, r2, r3, conservation, y


def compute_loss(net, t_pde, t_ic, w, causal_weight=None):
    r1, r2, r3, cons, y_pde = compute_residuals(net, t_pde)

    # Causal weighting: penalize later times less until early times are solved
    if causal_weight is not None:
        cw = causal_weight.to(t_pde.device)
        r1 = r1 * cw;  r2 = r2 * cw;  r3 = r3 * cw

    loss_r1   = (r1 ** 2).mean()
    loss_r2   = (r2 ** 2).mean()
    loss_r3   = (r3 ** 2).mean()
    loss_cons = (cons ** 2).mean()

    # IC loss
    y_ic = net(t_ic)
    ic_target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    # IC on y2 in log space is more stable
    loss_ic1 = (y_ic[:, 0:1] - 1.0) ** 2
    loss_ic2 = (torch.log10(y_ic[:, 1:2] + 1e-30) - torch.log10(torch.tensor(1e-30))) ** 2
    loss_ic3 = (y_ic[:, 2:3] - 0.0) ** 2
    loss_ic  = loss_ic1.mean() + loss_ic2.mean() + loss_ic3.mean()

    total = (w["r1"]   * loss_r1 +
             w["r2"]   * loss_r2 +
             w["r3"]   * loss_r3 +
             w["cons"] * loss_cons +
             w["ic"]   * loss_ic)

    return total, {
        "r1": loss_r1.item(), "r2": loss_r2.item(), "r3": loss_r3.item(),
        "cons": loss_cons.item(), "ic": loss_ic.item()
    }


# ── Causal weight schedule ────────────────────────────────────────────────────
def make_causal_weights(t_pde, epsilon=1.0):
    """
    Exponentially decaying weights from t=0 forward.
    Forces the network to get early times right before late times.
    """
    t_norm = t_pde / T_END  # [0, 1]
    w = torch.exp(-epsilon * t_norm)
    return w / w.max()


# ── Adaptive loss weight update (based on gradient magnitudes) ────────────────
def update_weights(net, t_pde, t_ic, w, alpha=0.9):
    """
    NTK-inspired rebalancing: scale each loss term so its gradient
    magnitude is roughly equal. Run every N iterations.
    """
    grads = {}
    for key in ["r1", "r2", "r3", "ic"]:
        w_test = {k: 0.0 for k in w}
        w_test[key] = 1.0
        loss, _ = compute_loss(net, t_pde, t_ic, w_test)
        loss.backward()
        total_grad = sum(
            p.grad.norm().item() ** 2
            for p in net.parameters() if p.grad is not None
        ) ** 0.5
        net.zero_grad()
        grads[key] = total_grad + 1e-10

    max_grad = max(grads.values())
    new_w = {k: alpha * w[k] + (1 - alpha) * (max_grad / grads[k])
             for k in ["r1", "r2", "r3", "ic"]}
    new_w["cons"] = w["cons"]
    return new_w


# ── Training setup ────────────────────────────────────────────────────────────
net = RobertsonNet(width=128, n_blocks=6).to(device)
t_ic = torch.tensor([[0.0]], dtype=torch.float32).to(device)

w = {"r1": 1.0, "r2": 1.0, "r3": 1.0, "cons": 10.0, "ic": 1000.0}

total_losses  = []
all_loss_info = []
phase_markers = []
total_time    = 0.0


# ── Curriculum: staged time windows ──────────────────────────────────────────
curriculum_stages = [
    {"t_end": 1.0,    "iters": 5000,  "lr": 2e-3, "causal_eps": 5.0},
    {"t_end": 5.0,    "iters": 5000,  "lr": 1e-3, "causal_eps": 3.0},
    {"t_end": 20.0,   "iters": 8000,  "lr": 5e-4, "causal_eps": 2.0},
    {"t_end": T_END,  "iters": 12000, "lr": 2e-4, "causal_eps": 1.0},
]

print("\n" + "=" * 60)
print("CURRICULUM ADAM TRAINING")
print("=" * 60)

for stage_idx, stage in enumerate(curriculum_stages):
    t_end_s   = stage["t_end"]
    n_iters   = stage["iters"]
    lr        = stage["lr"]
    causal_eps= stage["causal_eps"]

    t_pde = torch.linspace(0, t_end_s, 512, dtype=torch.float32).reshape(-1, 1).to(device)
    causal_w = make_causal_weights(t_pde, epsilon=causal_eps).reshape(-1, 1)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=lr*0.01)

    print(f"\n  Stage {stage_idx+1}: t ∈ [0, {t_end_s}] | {n_iters} iters | lr={lr}")
    phase_markers.append(len(total_losses))

    t0 = time.perf_counter()
    for it in range(n_iters):
        optimizer.zero_grad()

        # Adaptive weight update every 1000 iters
        if it % 1000 == 0 and it > 0:
            w = update_weights(net, t_pde, t_ic, w)

        loss, info = compute_loss(net, t_pde, t_ic, w, causal_weight=causal_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_losses.append(loss.item())
        all_loss_info.append(info)

        if (it + 1) % 2000 == 0:
            print(f"    iter {it+1:5d} | total: {loss.item():.3e} | "
                  f"r1: {info['r1']:.2e} r2: {info['r2']:.2e} "
                  f"r3: {info['r3']:.2e} ic: {info['ic']:.2e}")

    stage_time = time.perf_counter() - t0
    total_time += stage_time
    print(f"    Stage time: {stage_time:.1f}s")


# ── L-BFGS refinement on full domain ─────────────────────────────────────────
print("\n" + "=" * 60)
print("L-BFGS REFINEMENT (full domain)")
print("=" * 60)

t_pde_full = torch.linspace(0, T_END, 1024, dtype=torch.float32).reshape(-1, 1).to(device)
w_final    = {"r1": 1.0, "r2": 1.0, "r3": 1.0, "cons": 10.0, "ic": 1000.0}

optimizer_lbfgs = torch.optim.LBFGS(
    net.parameters(),
    lr=0.5,
    max_iter=100,
    max_eval=125,
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
    history_size=100,
    line_search_fn="strong_wolfe"
)

lbfgs_losses = []
phase_markers.append(len(total_losses))

t0 = time.perf_counter()
for lbfgs_round in range(20):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, info = compute_loss(net, t_pde_full, t_ic, w_final)
        loss.backward()
        lbfgs_losses.append(loss.item())
        total_losses.append(loss.item())
        return loss

    optimizer_lbfgs.step(closure)

    if (lbfgs_round + 1) % 5 == 0:
        last = lbfgs_losses[-1]
        print(f"  Round {lbfgs_round+1:2d} | loss: {last:.3e}")

lbfgs_time  = time.perf_counter() - t0
total_time += lbfgs_time
print(f"  L-BFGS time: {lbfgs_time:.1f}s | Total: {total_time:.1f}s")


# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION vs REFERENCE")
print("=" * 60)

t_test  = np.linspace(0, T_END, 2000)
y_ref   = ref.sol(t_test)

net.eval()
t_tensor = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred = net(t_tensor).cpu().numpy()

def l2_rel(pred, ref):
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-15)

for i, name in enumerate(["y1", "y2", "y3"]):
    err = l2_rel(y_pred[:, i], y_ref[i])
    print(f"  {name}: L2 relative error = {err:.4e}")

conservation_err = np.abs(y_pred.sum(axis=1) - 1.0).max()
print(f"  Max conservation error: {conservation_err:.4e}")


# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(17, 12))
labels = ["y₁", "y₂ (×10⁴)", "y₃"]
scales = [1.0, 1e4, 1.0]

for i in range(3):
    s       = scales[i]
    ref_i   = y_ref[i] * s
    pred_i  = y_pred[:, i] * s

    # Solution
    axes[i, 0].plot(t_test, ref_i,  "k-",  lw=2,   label="Reference (Radau)")
    axes[i, 0].plot(t_test, pred_i, "r--", lw=1.5, label="PINN")
    axes[i, 0].set_title(f"{labels[i]} — Solution")
    axes[i, 0].set_xlabel("t");  axes[i, 0].legend(fontsize=8);  axes[i, 0].grid(True)

    # Absolute error
    axes[i, 1].semilogy(t_test, np.abs(pred_i - ref_i) + 1e-20, "b-", lw=1.5)
    axes[i, 1].set_title(f"{labels[i]} — Absolute error")
    axes[i, 1].set_xlabel("t");  axes[i, 1].grid(True)

    # Relative error
    rel_err = np.abs(pred_i - ref_i) / (np.abs(ref_i) + 1e-15)
    axes[i, 2].semilogy(t_test, rel_err + 1e-20, "g-", lw=1.5)
    axes[i, 2].set_title(f"{labels[i]} — Relative error")
    axes[i, 2].set_xlabel("t");  axes[i, 2].grid(True)

# Loss history
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))

axes2[0].semilogy(total_losses, "b-", lw=0.8, alpha=0.8)
for pm in phase_markers:
    axes2[0].axvline(x=pm, color="gray", linestyle="--", alpha=0.6)
axes2[0].set_title("Total Loss History")
axes2[0].set_xlabel("Iteration");  axes2[0].set_ylabel("Loss");  axes2[0].grid(True)

# Individual loss components (subsample)
step = max(1, len(all_loss_info) // 500)
iters_sub = range(0, len(all_loss_info), step)
for key, color in [("r1","blue"),("r2","red"),("r3","green"),("ic","purple"),("cons","orange")]:
    axes2[1].semilogy(iters_sub,
                      [all_loss_info[j][key] for j in iters_sub],
                      color=color, lw=1.2, label=key)
axes2[1].set_title("Loss Components (Adam phases)")
axes2[1].set_xlabel("Iteration");  axes2[1].legend();  axes2[1].grid(True)

plt.suptitle(f"Robertson Stiff PINN — Improved Scratch | Total time: {total_time:.1f}s", fontsize=12)
plt.tight_layout()
plt.savefig("robertson_improved.png", dpi=150)
plt.show()
print("\nSaved robertson_improved.png")
