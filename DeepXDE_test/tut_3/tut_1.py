import os
import torch
import deepxde as dde

os.environ["DDE_BACKEND"] = "pytorch"

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = troch.device("cpu")
    print("MPS device not found, using CPU")

    torch.set_default_device(device)

    x = torch.zeros(1).to(device)
    print(f"Tensor is on: {x.device}")

    def pde(x,y):
        return dde.grad.hessian(y, x) + 1

    geom = dde.geometry.Interval(-1,1)
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on: on)
    data = dde.data.PDE(geom, pde, bc, 10, 2)
    net = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    model.train(epochs=10)
