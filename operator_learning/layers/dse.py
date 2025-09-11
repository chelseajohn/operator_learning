import torch
import torch.nn as nn
from operator_learning.utils.misc import format_complexTensor, deformat_complexTensor
from .linear import GridLinear

class SpectralConv_dse(nn.Module):
    def __init__(self, dv, transformer, kX, kY=None, bias=False, dim=2):
        super().__init__()
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        self.kX = kX       
        self.kY = kY
        self.channel = dv
        self.transformer = transformer

        self.scale = 1 / (dv * dv)
        if dim ==1:
            self.kX_sym = int(kX/2) + 1
            wr = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
            wi = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
            self.weightsReal = nn.Parameter(format_complexTensor(wr))
            self.weightsImag = nn.Parameter(format_complexTensor(wi))
        else:
            assert kY is not None, "kY must be specified for 2D simulations"
            wr = self.scale * torch.rand(dv, dv, kX , kY, dtype=torch.cfloat)
            wi = self.scale * torch.rand(dv, dv, kX , kY, dtype=torch.cfloat)
            self.weightsReal = nn.Parameter(format_complexTensor(wr))
            self.weightsImag = nn.Parameter(format_complexTensor(wi))

        if bias:
            init_std = (2/ (dv * dim))**0.5
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([dv]) + (1,) * dim))
            )
        else:
            self.bias = None
        
    
    def compl_mul(self, input, weights):
        """
        [batch, dv, nx, ny], [dv, dv, nx, ny] -> [batch, dv, nx, ny]
        Einsum string depends on dim:
        - 1D: "bik,iok->bok"
        - 2D: "bixy,ioxy->boxy"
        """
        R = deformat_complexTensor(weights).to(input.device)
        if self.dim == 1:
            return torch.einsum("bik,iok->bok", input, R)
        elif self.dim == 2:
            return torch.einsum("bixy,ioxy->boxy", input, R)
        else:
            raise ValueError("dim must be 1 or 2")

    def forward(self, x):
        b = x.shape[0]

        # Transform to fourier space
        x_ft = self.transformer.forward(x)  # Fourier coeffs (complex)

        if self.dim == 1:
            out_ft = torch.zeros(b, self.channel, self.kX, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.kX_sym] = self.compl_mul(x_ft[:, :, :self.kX_sym], self.weightsReal)
            out_ft[:, :, -self.kX_sym:] = self.compl_mul(x_ft[:, :, -self.kX_sym:], self.weightsImag)

        else:
            out_ft = torch.zeros(b, self.channel, 2*self.kX, self.kY, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.kX, :self.kY] = self.compl_mul(x_ft[:, :, :self.kX, :self.kY], self.weightsReal)
            out_ft[:, :, -self.kX:, :self.kY] = self.compl_mul(x_ft[:, :, -self.kX:, :self.kY], self.weightsImag)

        # Return to physical space
        x = self.transformer.inverse(out_ft)

        if self.bias is not None:
            x = x + self.bias

        return x


class DSELayer(nn.Module):
    def __init__(self,dv, 
                 transformer, 
                 kX, kY=None, 
                 non_linearity='gelu',
                 bias=False,
                 dim=2
                 ):
        super().__init__()

        
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)

       
        self.conv = SpectralConv_dse(dv, transformer, kX, kY, bias, dim)
        self.Wr = GridLinear(
                    inSize=dv, outSize=dv, hiddenSize=None,
                    bias=bias, n_layers=1, non_linearity=self.sigma,
                    n_dims=dim,
                    )
        self.Wi = GridLinear(inSize=dv, outSize=dv, hiddenSize=None,
                    bias=bias, n_layers=1, non_linearity=self.sigma,
                    n_dims=dim,
                    )           

    def forward(self, x):
        """ x[nBatch, dv, nY, nX] -> [nBatch, dv, nY, nX] """
        x = x.to(torch.cfloat)
        v = self.conv(x)
        w = self.Wr(x.real) + 1j * self.Wi(x.imag)
        v += w
        o = self.sigma(v.real) + 1j * self.sigma(v.imag)

        return o.real


