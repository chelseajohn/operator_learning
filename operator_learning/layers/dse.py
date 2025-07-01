
import numpy as np
import torch 
import torch.nn as nn
from .linear import GridLinear


class SpectralConv2d_dse (nn.Module):
    def __init__(self, dv, kX, kY, transformer):
        super().__init__()

        self.kX = kX       
        self.kY = kY
        self.channel = dv
        self.scale = (1 / (self.channel * self.channel))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.channel, self.channel, self.kX, self.kY, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.channel, self.channel, self.kX, self.kY, dtype=torch.cfloat))

        self.transformer = transformer

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # [batch, dv, nx,ny], [dv, dv, nx, ny] -> [batch, dv, nx, ny]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self.transformer.forward(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.channel,  2*self.kX, self.kY, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.kX, :self.kY] = self.compl_mul2d(x_ft[:, :, :self.kX, :self.kY], self.weights1)
        out_ft[:, :, -self.kX:, :self.kY] = self.compl_mul2d(x_ft[:, :, -self.kX:, :self.kY], self.weights2)
        
        # Return to physical space
        x = self.transformer.inverse(out_ft)
 
        return x


class DSELayer(nn.Module):
    def __init__(self, kX, kY, dv, 
                 transformer, 
                 non_linearity='gelu',
                 bias=False,
                 space_dim=2,
                 ):
        super().__init__()

        self.conv = SpectralConv2d_dse(dv, kX, kY, transformer)
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)

        self.Wr = GridLinear(inSize=dv,
                                outSize=dv,
                                hiddenSize=None,
                                bias=bias,
                                n_layers=1,
                                non_linearity=self.sigma,
                                space_dim=space_dim,
                                )
        self.Wi = GridLinear(inSize=dv,
                                outSize=dv,
                                hiddenSize=None,
                                bias=bias,
                                n_layers=1,
                                non_linearity=self.sigma,
                                space_dim=space_dim,
                                )

    def forward(self, x):
        """ x[nBatch, dv, nY, nX] -> [nBatch, dv, nY, nX] """

        v = self.conv(x)    # 2D Convolution
        w = self.Wr(x.real) + 1j *self.Wi(x.imag)      # Linear operator
        v += w
        o = self.sigma(v.real) + 1j * self.sigma(v.imag)
        
        return o
