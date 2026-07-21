import os, sys
import warnings
if os.getenv("ENABLE_FLOP_WRAPPERS", "0") == "1":
    from operator_learning.utils import flop_wrappers
import torch
import torch.nn as nn
from operator_learning.utils.communication import get_world_size, get_rank
from .linear import GridLinear
from .mlp import MLP
from operator_learning.utils.misc import (
    format_complexTensor,
    deformat_complexTensor,
    einsum_complexhalf,
    _dump_tensor, 
    ContiguousGrad
)

class SpectralConv_dse(nn.Module):
    def __init__(self, dv, kX, kY=None, kZ=None, dataClass='pic', bias=False, 
                 dim=1, use_complex_amp=False, tp_mesh=None, dtype=torch.complex128):
        
        """
        Spectral weights R and Bias must be kept in Float64 when particle 
        sharding is  enabled.
        """
        super().__init__()
        assert dim in (1,2, 3), "implemented only for PIC1D, PIC2D and PIC3D"
        self.dim = dim
        self.kX = kX     
        self.kY = kY if kY is not None else kX
        self.kZ = kZ if kZ is not None else kX
        self.channel = dv
        self.use_complex_amp = use_complex_amp
        self.data_type = dtype
        self.scale = 1 / (dv * dv)
        self.tp_mesh = tp_mesh
        if self.tp_mesh is not None:
            self.tp_size = self.tp_mesh.size()
        else:
            self.tp_size = 1
      
        if dim == 1:
            weights = self.scale * torch.rand(dv, dv, 2*self.kX, dtype=self.data_type)
            self.R = nn.Parameter(format_complexTensor(weights))
        elif dim == 2:
            weights = self.scale * torch.rand(dv, dv, 2*self.kX, 2*self.kY, dtype=self.data_type)
            self.R = nn.Parameter(format_complexTensor(weights))
        else:
            weights = self.scale * torch.rand(dv, dv, 2*self.kX, 2*self.kY, 2*self.kZ, dtype=self.data_type)
            self.R = nn.Parameter(format_complexTensor(weights))

        if bias:
            init_std = (2/(dv * dim))**0.5
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([dv]) + (1,)))
            ).to(torch.float64)
        else:
            self.bias = None
        
    
    def compl_mul(self, input, weights):
        """
        PIC1D: input[batchsize, dv, kX], weights[dv, dv, kX]
        PIC2D: input[batchsize, dv, kX, kY], weights[dv, dv, kX, kY]
        PIC3D: input[batchsize, dv, kX, kY, kZ], weights[dv, dv, kX, kY, kZ]
        Returns PIC1D: [batchsize, dv, kX]
        Returns PIC2D: [batchsize, dv, kX, kY]
        Returns PIC3D: [batchsize, dv, kX, kY, kZ]
        """
        
        if self.training and self.use_complex_amp:
            einsum_fn = einsum_complexhalf
            R = deformat_complexTensor(weights.half()).to(input.device)  # complex32
        else:
            einsum_fn = torch.einsum
            R = deformat_complexTensor(weights).to(input.device) # complex128

        if self.dim == 1:
            return einsum_fn("bik,iok->bok", input, R)
        elif self.dim == 2:
            return einsum_fn("bixy,ioxy->boxy", input, R)
        else:
            return einsum_fn("bixyz,ioxyz->boxyz", input, R)

        
    def forward(self, x, transform):
        """
        PIC 1D/2D/3D:  x[batchsize, dv, nParticle], 
        returns: [batchsize, dv, nParticle]
        """

        if self.training and self.use_complex_amp:
            dtype = torch.complex32 
        else:
            dtype = self.data_type
        batchsize = x.shape[0]

        # print(f'x: {x.dtype}, {dtype}', flush=True)
        # _dump_tensor("x", x)

        # Transform to fourier space
        # Fourier coeffs (complex)
        x_ft = transform.forward(x.to(dtype))  # [batchsize, dv, modes]
        x_ft = x_ft.contiguous()


        # print(f'x_ft: {x_ft.dtype}', flush=True)
        # _dump_tensor("x_ft", x_ft)


        if self.tp_size > 1:
            # torch.distributed.all_reduce(x_ft, group=self.tp_mesh.get_group())
            x_ft = torch.distributed.nn.functional.all_reduce(x_ft.contiguous(), 
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=self.tp_mesh.get_group())
            x_ft =  ContiguousGrad.apply(x_ft)

            # print(f'x_ft_after: {x_ft.dtype}', flush=True)
            # _dump_tensor("x_ft_after", x_ft)

            
        if self.dim == 1:
            out_ft = self.compl_mul(x_ft, self.R)  # [batchsize, dv, kX]
        elif self.dim == 2:
            x_ft = torch.reshape(x_ft, (batchsize, self.channel, 2*self.kX, 2*self.kY))  # [batchsize, dv, 2*kX, 2*kY]
            out_ft = self.compl_mul(x_ft, self.R)
            out_ft = torch.reshape(out_ft, (batchsize, self.channel, 2*self.kX*(2*self.kY)))  # [batchsize, dv, modes]
        else:
            x_ft = torch.reshape(x_ft, (batchsize, self.channel, 2*self.kX, 2*self.kY, 2*self.kZ))  # [batchsize, dv, 2*kX, 2*kY, 2*kZ]
            out_ft = self.compl_mul(x_ft, self.R)
            out_ft = torch.reshape(out_ft, (batchsize, self.channel, 2*self.kX*(2*self.kY)*(2*self.kZ)))  # [batchsize, dv, modes]
        
        # print(f"out_ft: {out_ft.dtype}", flush=True)
        # _dump_tensor("out_ft", out_ft)

        # Return to physical space
        x  = transform.inverse(out_ft)   # [batchsize, dv, nParticle]
       
        # _dump_tensor("x_inv", x)
        # print(f'x_inv: {x.dtype}', flush=True)

        x = (x / (x.size(-1) * self.tp_size)) * self.dim 
        
        # _dump_tensor("x_out", x)

        if self.bias is not None:
            x = x + self.bias
        # _dump_tensor("x_bias", x)

        return x.real


class DSELayer(nn.Module):
    def __init__(self,dv, 
                 kX, kY=None, kZ=None,
                 dataClass='pic',
                 non_linearity='gelu',
                 bias=False,
                 dim=1,
                 use_complex_amp=False,
                 tp_mesh=None,
                 dtype=torch.float64
                 ):
        super().__init__()

        
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)
        
        self.tp_mesh = tp_mesh
        self.conv = SpectralConv_dse(dv=dv, kX=kX, kY=kY, kZ=kZ, 
                                    dataClass=dataClass,
                                    bias=bias, dim=dim,
                                    use_complex_amp=use_complex_amp,
                                    tp_mesh=tp_mesh,
                                    dtype=torch.complex128)
    
        # self.W = GridLinear(
        #                inSize=dv, outSize=dv, hiddenSize=None,
        #                bias=bias, n_layers=1, non_linearity=self.sigma,
        #                n_dims=1, 
        #                )

        self.W = MLP(mode='channel',
                     n_dims=1,
                     n_layers=1,
                     in_channels=dv,
                     out_channels=dv,
                     hidden_channels=None,
                     dtype=dtype,   # Need float64 when particle sharding is enabled
                    )
 
    def forward(self, x, transform):
        """
        PIC1D/2D/3D: x[batchsize, dv, nParticle]
        Returns: [batchsize, dv, nParticle]
        """
        # _dump_tensor(f"x_init", x)
        v = self.conv(x, transform)
        # _dump_tensor("v",v)

        w = self.W(x)
        # _dump_tensor("w",w)

        o = self.sigma(v+w)
        # _dump_tensor("o",o)

        return o


