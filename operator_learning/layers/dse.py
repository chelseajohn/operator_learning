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
                 dim=1, use_complex_amp=False, device_mesh=None):
        super().__init__()
        assert dim in (1,2), "implemented only for PIC1D and PIC2D"
        self.dim = dim
        self.kX = kX     
        self.kY = kY if kY is not None else kX
        self.channel = dv
        self.use_complex_amp = use_complex_amp
        self.scale = 1 / (dv * dv)
        self.device_mesh = device_mesh
        if device_mesh is not None and "tp" in device_mesh.mesh_dim_names:
            self.TP_enabled = True
            self.tp_size = device_mesh["tp"].size()
        else:
            self.TP_enabled = False
            self.tp_size = 1
      

        if dim == 1:
            weights = self.scale * torch.rand(dv, dv, 2*self.kX, dtype=torch.cfloat)
            self.R = nn.Parameter(format_complexTensor(weights))
        else:
            weights = self.scale * torch.rand(dv, dv, 2*self.kX, 2*self.kY, dtype=torch.cfloat)
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
        x_ft = transform.forward(x.to(dtype))  # [batchsize, dv, modes]

        if self.TP_enabled:
            torch.distributed.all_reduce(x_ft, group=self.device_mesh.get_group())
    
        if self.dim == 1:
            out_ft = self.compl_mul(x_ft, self.R)  # [batchsize, dv, kX]
        else:
            x_ft = torch.reshape(x_ft, (batchsize, self.channel, 2*self.kX, 2*self.kY))  # [batchsize, dv, 2*kX, 2*kY]
            out_ft = self.compl_mul(x_ft, self.R)
            out_ft = torch.reshape(out_ft, (batchsize, self.channel, 2*self.kX*(2*self.kY)))  # [batchsize, dv, 2*kX, 2*kY]
     

        # Return to physical space
        x  = transform.inverse(out_ft) # [batchsize, dv, nParticle]
        x = x / x.size(-1) * self.dim 

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
                 device_mesh=None,
                 ):
        super().__init__()

        
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)
        
        self.device_mesh = device_mesh
        self.conv = SpectralConv_dse(dv, kX, kY, dataClass, bias, dim, use_complex_amp, device_mesh)
    
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


