import os
if os.getenv("ENABLE_FLOP_WRAPPERS", "0") == "1":
    from operator_learning.utils import flop_wrappers
import torch
import torch.nn as nn
from operator_learning.utils.misc import format_complexTensor, deformat_complexTensor, einsum_complexhalf
from .linear import GridLinear
from .mlp import MLP

class SpectralConv_nufft(nn.Module):
    def __init__(self, dv, kX, dataClass='pic', bias=False, dim=1, use_complex_amp=False):
        super().__init__()
        assert dim == 1, "supported only for PIC1D layout"
        self.dim = dim
        self.kX = kX     
        self.channel = dv
        self.use_complex_amp = use_complex_amp
        self.scale = 1 / (dv * dv)
      
        weights = self.scale * torch.rand(dv, dv, kX, dtype=torch.cfloat)
        self.R = nn.Parameter(format_complexTensor(weights))

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
        Returns PIC1D: [batchsize, dv, kX]
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
        PIC1D:  x[batchsize, dv, nParticle], 
        returns: [batchsize, dv, nParticle]
        """

        if self.training and self.use_complex_amp:
            dtype = torch.complex32 
        else:
            dtype = torch.cfloat

        # Transform to fourier space
        x_ft = transform.forward(x.to(dtype))  # [batchsize, dv, nParticle]

        out_ft = self.compl_mul(x_ft, self.R)  # [batchsize, dv, kX]
       
        # Return to physical space
        x  = transform.inverse(out_ft) # [batchsize, dv, nParticle]
        x = x / x.size(-1) * self.dim  # [batchsize, dv, nParticle]

        if self.bias is not None:
            x = x + self.bias

        return x.real


class NUFFTLayer(nn.Module):
    def __init__(self,dv, 
                 kX, 
                 dataClass='pic',
                 non_linearity='gelu',
                 bias=False,
                 dim=1,
                 use_complex_amp=False
                 ):
        super().__init__()

        
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)

        self.conv = SpectralConv_nufft(dv, kX, dataClass, bias, dim, use_complex_amp)
        
        # self.W = GridLinear(
        #                 inSize=dv, outSize=dv, hiddenSize=None,
        #                 bias=bias, n_layers=1, non_linearity=self.sigma,
        #                 n_dims=1, 
        #                 )
        self.W = MLP( mode='channel',
                        n_dims=1,
                        n_layers=1,
                        in_channels=dv,
                        out_channels=dv,
                        hidden_channels=None,
                    )
        

    def forward(self, x, transform):
        """ x[nBatch, dv, nParticle] -> [nBatch, dv, nparticle] """
        
        v = self.conv(x, transform)
        w = self.W(x)
        o = self.sigma(v+w)

        return o


