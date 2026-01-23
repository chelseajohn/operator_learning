import os
if os.getenv("ENABLE_FLOP_WRAPPERS", "0") == "1":
    from operator_learning.utils import flop_wrappers
import torch
import torch.nn as nn
from operator_learning.utils.misc import format_complexTensor, deformat_complexTensor, einsum_complexhalf
from .linear import GridLinear

class SpectralConv_nufft(nn.Module):
    def __init__(self, dv, transformer, kX, dataClass='pic', bias=False, dim=1, use_complex_amp=False):
        super().__init__()
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        self.kX = kX       
        self.channel = dv
        self.transformer = transformer
        self.use_complex_amp = use_complex_amp
        
        self.scale = 1 / (dv * dv)
        weights1 = self.scale * torch.rand(dv, dv, kX, dtype=torch.cfloat)
        self.R1 = nn.Parameter(format_complexTensor(weights1))

        if bias:
            init_std = (2/ dv)**0.5
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([dv]) + (1,)))
            )
        else:
            self.bias = None
        
    
    def compl_mul(self, input, weights):
        """
        [batch, dv, nParticle], [dv, dv, nParticle] -> [batch, dv, nParticle]
        Einsum string : "bik,iok->bok"
        """
        
        if self.training and self.use_complex_amp:
            einsum_fn = einsum_complexhalf
            R = deformat_complexTensor(weights.half()).to(input.device)  # complex32
        else:
            einsum_fn = torch.einsum
            R = deformat_complexTensor(weights).to(input.device) # complex64

        return einsum_fn("bik,iok->bok", input, R)
       

    def forward(self, x):
        b = x.shape[0]

        if self.training and self.use_complex_amp:
            dtype = torch.complex32 
        else:
            dtype = torch.cfloat

        # Transform to fourier space
        x_ft = self.transformer.forward(x.to(dtype))  # Fourier coeffs (complex)
        print(f'x_ft: {x_ft.shape}')

        out_ft = torch.zeros(b, self.channel, x.shape[-1], dtype=dtype, device=x.device)
        out_ft[:, :, :self.kX] = self.compl_mul(x_ft[:, :, :self.kX], self.R1)
       
        # Return to physical space
        x = self.transformer.inverse(out_ft)

        if self.bias is not None:
            x = x + self.bias
        print(f'x: {x.shape}')
        return x.real


class NUFFTLayer(nn.Module):
    def __init__(self,dv, 
                 transformer, 
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

        self.conv = SpectralConv_nufft(dv, transformer, kX, dataClass, bias, dim, use_complex_amp)
        if dataClass == 'pic':
            dim = 1 # same execution as 1D since y-cord is a channel
        self.W = GridLinear(
                        inSize=dv, outSize=dv, hiddenSize=None,
                        bias=bias, n_layers=1, non_linearity=self.sigma,
                        n_dims=dim, 
                        )
        

    def forward(self, x):
        """ x[nBatch, dv, nParticle] -> [nBatch, dv, nparticle] """
        v = self.conv(x)
        w = self.W(x)
        o = self.sigma(v+w)

        return o


