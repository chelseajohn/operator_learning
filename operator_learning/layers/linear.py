import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

class GridLinear(nn.Module):
    """
    Custom Linear Layer with torch.einsum
    """

    def __init__(self, inSize, outSize, hiddenSize=None, n_dims=2,
                 bias=False, n_layers=1, non_linearity=F.gelu):
        super().__init__()

        assert n_dims in (2, 3), "spatial dimension must be 2 or 3"
        self.n_dims = n_dims
        self.n_layers = n_layers
        hiddenSize = outSize if hiddenSize is None else hiddenSize 
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList() if bias else None
        layer_sizes = [inSize] + [hiddenSize] * (n_layers - 1) + [outSize]
        self.non_linearity = non_linearity

        for i in range(n_layers):
            self.weights.append(nn.Parameter(torch.empty(layer_sizes[i + 1], layer_sizes[i])))
            if bias:
                if n_dims == 2:
                    self.biases.append(nn.Parameter(torch.empty(layer_sizes[i + 1], 1, 1)))
                else:
                    self.biases.append(nn.Parameter(torch.empty(layer_sizes[i + 1], 1, 1, 1)))

        

        # Initialize parameters (same as in pytorch for nn.Linear)
        for i,weight in enumerate(self.weights):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.biases[i], -bound, bound)

    def forward(self, x):
        """ x[nBatch, inSize, nX, nY, (nZ)] -> [nBatch, outSize, nX, nY, (nZ)] """

        if self.n_dims == 2:
            einsum_str = "ij,ejxy->eixy"
        else:
            einsum_str = "ij,ejxyz->eixyz"

        for i in range(self.n_layers):
            x = torch.einsum(einsum_str, self.weights[i], x)
            if self.biases is not None:
                x += self.biases[i]
            if i < self.n_layers - 1:
                x = self.non_linearity(x)

        return x