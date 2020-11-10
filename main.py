import numpy as np

import torch

from torch import Tensor
from torch.nn import Module, Linear, ReLU, Parameter

from captum.attr import IntegratedGradients

class ToyModel(Module):
    def __init__(self):
        super().__init__()
        self.lin1: Linear = Linear(3, 3)
        self.lin2: Linear = Linear(3, 2)
        self.relu: ReLU = ReLU()

        self.lin1.weight: Parameter = Parameter(torch.arange(-4.0, 5.0).view(3, 3))
        self.lin1.bias: Parameter = Parameter(torch.zeros(1, 3))
        self.lin2.weight: Parameter = Parameter(torch.arange(-3.0, 3.0).view(2, 3))
        self.lin2.bias: Parameter = Parameter(torch.ones(1, 2))

    def forward(self, input: Tensor) -> Linear:
        return self.lin2(self.relu(self.lin1(input)))

model: ToyModel = ToyModel()
model.eval()

torch.manual_seed(123)
np.random.seed(123)

input: Tensor = torch.rand(2, 3)
baseline: Tensor = torch.zeros(2, 3)

ig: IntegratedGradients = IntegratedGradients(model)

# Union[Tensor, Tuple[Tensor, ...]
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
