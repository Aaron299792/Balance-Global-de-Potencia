import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import torch
import time


def train_timed(model, iterations, device):
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    losshistory, train_state = model.train(iterations=iterations)

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    print(f"[{device.type.upper()}] Training time: {elapsed:.2f}s ({iterations} iterations, {elapsed/iterations*1000:.2f}ms/iter)")
    return losshistory, train_state, elapsed


def build_model(device):
    torch.set_default_device(device)

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t)
        d2y_dt2 = dde.grad.hessian(y, t)
        return d2y_dt2 - 10 * dy_dt + 9 * y - 5 * t

    def func(t):
        return 50 / 81 + t * 5 / 9 - 2 * np.exp(t) + (31 / 81) * np.exp(9 * t)

    geom = dde.geometry.TimeDomain(0, 0.25)

    def boundary_l(t, on_initial):
        return on_initial and dde.utils.isclose(t[0], 0)

    def bc_func2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 2

    ic1 = dde.icbc.IC(geom, lambda x: -1, lambda _, on_initial: on_initial)
    ic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

    data = dde.data.TimePDE(geom, ode, [ic1, ic2], 16, 2, solution=func, num_test=500)
    net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot uniform")

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[0.01, 1, 1])
    model.net.to(device)
    return model


ITERATIONS = 10000
dde.config.set_default_float("float32")

results = {}

# --- CPU Run ---
print("=" * 50)
print("Running on CPU...")
print("=" * 50)
cpu_device = torch.device("cpu")
cpu_model = build_model(cpu_device)
_, _, cpu_time = train_timed(cpu_model, ITERATIONS, cpu_device)
results["CPU"] = cpu_time

# --- MPS Run ---
if torch.backends.mps.is_available():
    print("\n" + "=" * 50)
    print("Running on MPS (Apple GPU)...")
    print("=" * 50)
    mps_device = torch.device("mps")
    mps_model = build_model(mps_device)
    _, _, mps_time = train_timed(mps_model, ITERATIONS, mps_device)
    results["MPS"] = mps_time
else:
    print("\nMPS not available, skipping GPU run.")

# --- Summary ---
print("\n" + "=" * 50)
print("BENCHMARK SUMMARY")
print("=" * 50)
for device_name, elapsed in results.items():
    print(f"  {device_name}: {elapsed:.2f}s ({elapsed/ITERATIONS*1000:.2f}ms/iter)")

if "MPS" in results and "CPU" in results:
    speedup = results["CPU"] / results["MPS"]
    if speedup > 1:
        print(f"\n  MPS was {speedup:.2f}x FASTER than CPU")
    else:
        print(f"\n  CPU was {1/speedup:.2f}x FASTER than MPS")
print("=" * 50)
