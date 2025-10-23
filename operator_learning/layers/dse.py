import os
if os.getenv("ENABLE_FLOP_WRAPPERS", "0") == "1":
    from operator_learning.utils import flop_wrappers
import torch
import torch.nn as nn
from operator_learning.utils.misc import format_complexTensor, deformat_complexTensor, einsum_complexhalf
from .linear import GridLinear

class SpectralConv_dse(nn.Module):
    def __init__(self, dv, transformer, kX, kY=None, dataClass='pic', bias=False, dim=2):
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
            weights1 = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
            weights2 = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
            self.R1 = nn.Parameter(format_complexTensor(weights1))
            self.R2 = nn.Parameter(format_complexTensor(weights2))
        else:
            assert kY is not None, "kY must be specified for 2D simulations"
            weights1 = self.scale * torch.rand(dv, dv, kX , kY, dtype=torch.cfloat)
            weights2 = self.scale * torch.rand(dv, dv, kX , kY, dtype=torch.cfloat)
            self.R1 = nn.Parameter(format_complexTensor(weights1))
            self.R2 = nn.Parameter(format_complexTensor(weights2))

        if bias:
            init_std = (2/ (dv * dim))**0.5
            if dataClass == 'pic':
                dim = 1
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
        use_complexhalf = (R.dtype == torch.complex32 and input.dtype == torch.complex32)
        einsum_fn = einsum_complexhalf if use_complexhalf else torch.einsum

        if self.dim == 1:
            return einsum_fn("bik,iok->bok", input, R)
        elif self.dim == 2:
            return einsum_fn("bixy,ioxy->boxy", input, R)
        else:
            raise ValueError("dim must be 1 or 2")

    def forward(self, x):
        b = x.shape[0]

        # Transform to fourier space
        x_ft = self.transformer.forward(x.cfloat())  # Fourier coeffs (complex)

        if self.dim == 1:
            out_ft = torch.zeros(b, self.channel, self.kX, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.kX_sym] = self.compl_mul(x_ft[:, :, :self.kX_sym], self.R1)
            out_ft[:, :, -self.kX_sym:] = self.compl_mul(x_ft[:, :, -self.kX_sym:], self.R2)

        else:
            # ToDO: fix for RBC
            # Reshape into (kx, ky) frequency grid
            x_ft = x_ft.view(b, self.channel, 2 * self.kX, 2 * self.kY - 1)
            out_ft = x_ft.new_empty((b, self.channel, 2 * self.kX, self.kY))
            out_ft[:, :, :self.kX, :self.kY] = self.compl_mul(x_ft[:, :, :self.kX, :self.kY], self.R1)
            out_ft[:, :, -self.kX:, :self.kY] = self.compl_mul(x_ft[:, :, -self.kX:, :self.kY], self.R2)
            out_ft = out_ft.flatten(2)  # [b, c, 2*kX*kY]
            # Take advantage of real input data and the FFT has complex conjugate symmetry and hence the flip and conj
            out_ft = torch.cat(
                [out_ft, out_ft[:, :, 2 * self.kX:].conj().flip(-1)], dim=-1
            )

        # Return to physical space
        x = self.transformer.inverse(out_ft)

        if self.bias is not None:
            x = x + self.bias

        return x.real


class DSELayer(nn.Module):
    def __init__(self,dv, 
                 transformer, 
                 kX, kY=None, 
                 dataClass='pic',
                 non_linearity='gelu',
                 bias=False,
                 dim=1
                 ):
        super().__init__()

        
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)

        self.conv = SpectralConv_dse(dv, transformer, kX, kY, dataClass, bias, dim)
        if dataClass == 'pic':
            dim = 1 # same execution as 1D since y-cord is a channel
        self.W = GridLinear(
                        inSize=dv, outSize=dv, hiddenSize=None,
                        bias=bias, n_layers=1, non_linearity=self.sigma,
                        n_dims=dim, 
                        )
        

    def forward(self, x):
        """ x[nBatch, dv, nX, nY] -> [nBatch, dv, nX, nY] """
        v = self.conv(x)
        w = self.W(x)
        v += w
        o = self.sigma(v)

        return o


