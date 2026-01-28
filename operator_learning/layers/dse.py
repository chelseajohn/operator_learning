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
        self.dim = dim
        self.kX = kX     
        self.kY = kY if kY is not None else kX
        self.channel = dv
        self.use_complex_amp = use_complex_amp
        self.scale = 1 / (dv * dv)
      
        # self.kX_sym = int(kX/2) + 1
        # self.kX_sym = int(kX/2)
        # weights1 = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
        # weights2 = self.scale * torch.rand(dv, dv, self.kX_sym, dtype=torch.cfloat)
        # self.R1 = nn.Parameter(format_complexTensor(weights1))
        # self.R2 = nn.Parameter(format_complexTensor(weights2))

        if dim == 1:
            weights = self.scale * torch.rand(dv, dv, self.kX, dtype=torch.cfloat)
            self.R = nn.Parameter(format_complexTensor(weights))
        else:
            weights1 = self.scale * torch.rand(dv, dv, kX, kY, dtype=torch.cfloat)
            weights2 = self.scale * torch.rand(dv, dv, kX , kY, dtype=torch.cfloat)
            self.R1 = nn.Parameter(format_complexTensor(weights1))
            self.R2 = nn.Parameter(format_complexTensor(weights2))

        if bias:
            init_std = (2/(dv * dim))**0.5
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([dv]) + (1,)))
            )
        else:
            self.bias = None
        
    
    def compl_mul(self, input, weights):
        """
        PIC1D: input[batchsize, dv, kX], weights[dv, dv, kX]
        PIC2D: input[batchsize, dv, kX, kY], weights[dv, dv, kX, kY]
        Returns PIC1D: [batchsize, dv, kX]
        Returns PIC2D: [batchsize, dv, kX, kY]
        """
        
        if self.training and self.use_complex_amp:
            einsum_fn = einsum_complexhalf
            R = deformat_complexTensor(weights.half()).to(input.device)  # complex32
        else:
            einsum_fn = torch.einsum
            R = deformat_complexTensor(weights).to(input.device) # complex64

        if self.dim == 1:
            return einsum_fn("bik,iok->bok", input, R)
        else:
            return einsum_fn("bixy,ioxy->boxy", input, R)

        
    def forward(self, x, transform):
        """
        PIC1D/2D:  x[batchsize, dv, nParticle], 
        returns: [batchsize, dv, nParticle]
        """

        if self.training and self.use_complex_amp:
            dtype = torch.complex32 
        else:
            dtype = torch.cfloat

        batchsize = x.shape[0]

        # Transform to fourier space
        # Fourier coeffs (complex)
        x = x.permute(0, 2, 1)  # [batchsize, nParticle, dv]
        x_ft = transform.forward(x.to(dtype))  # [batchsize, modes, dv]
        x_ft = x_ft.permute(0, 2, 1) # [batchsize, dv, modes]

        # out_ft = torch.zeros(b, self.channel, self.kX, dtype=dtype, device=x.device)
        # out_ft[:, :, :self.kX_sym] = self.compl_mul(x_ft[:, :, :self.kX_sym], self.R1)
        # out_ft[:, :, -self.kX_sym:] = self.compl_mul(x_ft[:, :, -self.kX_sym:], self.R2)

        if self.dim == 1:
            out_ft = self.compl_mul(x_ft, self.R)  # [batchsize, dv, modes]
        else:
            x_ft = torch.reshape(x_ft, (batchsize, self.channel, 2*self.kX, 2*self.kY-1))  # [batchsize, dv, 2*kX, 2*kY-1]
            out_ft = torch.zeros(batchsize, self.channel, 2*self.kX, self.kY, dtype=dtype, device=x.device) #[batchsize, dv, 2*kX, kY]
            out_ft[:, :, :self.kX, :self.kY] = self.compl_mul(x_ft[:, :, :self.kX, :self.kY], self.R1)
            ## Seems weight1 and weight2 are for positive and negative modes (in x or first dimension) but why different weights?
            out_ft[:, :, -self.kX:, :self.kY] = self.compl_mul(x_ft[:, :, -self.kX:, :self.kY], self.R2) 
            x_ft1 = torch.reshape(out_ft, (batchsize, self.channel, 2*self.kX*self.kY))  # [batchsize, dv, 2*kX*kY]
            
            ## Take advantage of real input data and the FFT has complex conjugate symmetry and hence the flip and conj
            if dtype == torch.cfloat:
                x_ft2 = x_ft1[..., 2 * self.kX:].flip(-1, -2).conj()
            else:
                # flip not supported for complexHalf
                x_ft2 = x_ft1[..., 2 * self.kX:].to(torch.complex64).flip(-1, -2).conj().to(out_ft.dtype)

            out_ft = torch.cat([x_ft1, x_ft2], dim=-1) # [batchsize, dv, 2*kX*(2*kY - 1)]
     

        out_ft = out_ft.permute(0, 2, 1) # [batchsize, modes, dv]
        
        # Return to physical space
        x  = transform.inverse(out_ft) # [batchsize, nParticle, dv]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * self.dim  # [batchsize, dv, nParticle]

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


