import os
import warnings
if os.getenv("ENABLE_FLOP_WRAPPERS", "0") == "1":
    from operator_learning.utils import flop_wrappers
import torch
import torch.nn as nn
from operator_learning.utils.misc import format_complexTensor, deformat_complexTensor, einsum_complexhalf
from .linear import GridLinear
from .mlp import MLP

class SpectralConv_dse(nn.Module):
    def __init__(self, dv, kX, kY=None, dataClass='pic', bias=False, 
                 dim=1, use_complex_amp=False):
        super().__init__()
        assert dim in (1,2), "implemented only for PIC1D and PIC2D"
        self.kX = kX     
        if kY is not None:
            warnings.warn("PIC uses only a 1D implementation; kY is ignored.", RuntimeWarning)
        self.channel = dv
        self.use_complex_amp = use_complex_amp
        self.scale = 1 / (dv * dv)
      
        # self.kX_sym = int(kX/2) + 1
        # self.kX_sym = int(kX/2)
        # weights1 = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
        # weights2 = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
        # self.R1 = nn.Parameter(format_complexTensor(weights1))
        # self.R2 = nn.Parameter(format_complexTensor(weights2))

        weights = self.scale * torch.rand(dv, dv, self.kX, dtype=torch.cfloat)
        self.R = nn.Parameter(format_complexTensor(weights))
        
        if bias:
            init_std = (2/ dv)**0.5
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([dv]) + (1,)))
            )
        else:
            self.bias = None
        
    
    def compl_mul(self, input, weights):
        """
        PIC1D/2D: input[batchsize, dv, modes]
                  weights[dv, dv, modes]
        Returns: [batchsize, dv, modes]
        """
        
        if self.training and self.use_complex_amp:
            einsum_fn = einsum_complexhalf
            R = deformat_complexTensor(weights.half()).to(input.device)  # complex32
        else:
            einsum_fn = torch.einsum
            R = deformat_complexTensor(weights).to(input.device) # complex64
        return einsum_fn("bik,iok->bok", input, R)

        

    def forward(self, x, transform):
        """
        PIC1D/2D:  x[batchsize, dv, nParticle], 
        returns: [batchsize, dv, nParticle]
        """

        if self.training and self.use_complex_amp:
            dtype = torch.complex32 
        else:
            dtype = torch.cfloat

        # Transform to fourier space
        # Fourier coeffs (complex)
        x = x.permute(0, 2, 1)  # [batchsize, nParticle, dv]
        x_ft = transform.forward(x.to(dtype))  # [batchsize, modes, dv]
        x_ft = x_ft.permute(0, 2, 1) # [batchsize, dv, modes]

        # out_ft = torch.zeros(b, self.channel, self.kX, dtype=dtype, device=x.device)
        # out_ft[:, :, :self.kX_sym] = self.compl_mul(x_ft[:, :, :self.kX_sym], self.R1)
        # out_ft[:, :, -self.kX_sym:] = self.compl_mul(x_ft[:, :, -self.kX_sym:], self.R2)

        out_ft = self.compl_mul(x_ft, self.R)  # [batchsize, dv, modes]
        out_ft = out_ft.permute(0, 2, 1) # [batchsize, modes, dv]
                
        # Return to physical space
        x  = transform.inverse(out_ft) # [batchsize, nParticle, dv]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1)  # [batchsize, dv, nParticle]

        if self.bias is not None:
            x = x + self.bias

        return x.real


class DSELayer(nn.Module):
    def __init__(self,dv, 
                 kX, kY=None,
                 dataClass='pic',
                 non_linearity='gelu',
                 bias=False,
                 dim=1,
                 use_complex_amp=False,
                 ):
        super().__init__()

        
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)

        self.conv = SpectralConv_dse(dv, kX, kY, dataClass, bias, dim, use_complex_amp)
    
        # self.W = GridLinear(
        #                inSize=dv, outSize=dv, hiddenSize=None,
        #                bias=bias, n_layers=1, non_linearity=self.sigma,
        #                n_dims=1, 
        #                )

        self.W = MLP( mode='channel',
                        n_dims=1,
                        n_layers=1,
                        in_channels=dv,
                        out_channels=dv,
                        hidden_channels=None,
                    )
 
    def forward(self, x, transform):
        """
        PIC1D/2D: x[batchsize, dv, nParticle]
        Returns: [batchsize, dv, nParticle]
        """
        v = self.conv(x, transform)
        w = self.W(x)
        o = self.sigma(v+w)

        return o


