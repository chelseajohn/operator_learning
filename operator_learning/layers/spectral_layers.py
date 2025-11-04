import os
if os.getenv("ENABLE_FLOP_WRAPPERS", "0") == "1":
    from operator_learning.utils import flop_wrappers
import torch
import torch.nn as nn
from operator_learning.utils.misc import format_complexTensor, deformat_complexTensor, einsum_complexhalf

class SpectralConv(nn.Module):
    """
    Spectral convolution with FFT, linear transform, and Inverse FFT. 

    Args:
        dv (int): channels
        kX, kY, kZ (int): Fourier modes 
        bias (bool): bias for Fourier layer. Default is False.
        dim (int): spatial dim (1, 2 or 3). Default is 2.

  
    """
    def __init__(self, 
                 dv, 
                 kX, 
                 kY=None,
                 kZ=None,
                 bias=False,
                 dim=2,
                 use_complex_amp=False
    ):
        super().__init__()

        assert dim in (1, 2, 3), "Only till 3D supported"
        self.dim = dim
        self.dv = dv
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        # k_max = 12 in http://arxiv.org/pdf/2010.08895
        self.kX = kX           
        self.kY = kY
        self.kZ = kZ 
        self.use_complex_amp = use_complex_amp

        # R must be in torch.cfloat for torch.fft
        if dim ==1:
            operator_weights = torch.rand(dv, dv, kX, dtype=torch.cfloat)

        elif dim == 2:
            assert kY is not None, "kY must be specified for 2D simulations"
            operator_weights = torch.rand(dv, dv, 2 * kX , kY, dtype=torch.cfloat)
            
        else:
            assert kZ is not None, "kZ must be specified for 3D"
            operator_weights = torch.rand(dv, dv, 2 * kX, 2 * kY, kZ, dtype=torch.cfloat)
        self.R = nn.Parameter(format_complexTensor(operator_weights))


        if bias:
            init_std = (2/ (dv * dim))**0.5
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([dv]) + (1,) * dim))
            )
        else:
            self.bias = None

 
    def T(self, kMax, n, device, sym=False):
        if self.training and self.use_complex_amp:
            dtype = torch.complex32
        else:
            dtype = torch.cfloat

        T = torch.cat([
            torch.eye(kMax, dtype=dtype, device=device),                  # Top-left identity
            torch.zeros(kMax, n - kMax, dtype=dtype, device=device)       # Zero-pad to match n columns
        ], dim=1)                                                         # Shape: [kMax, n]

        if sym:
            Tinv = torch.cat([
                torch.zeros(kMax, n - kMax, dtype=dtype, device=device),   # Zero-pad on the left
                torch.eye(kMax, dtype=dtype, device=device)                # Bottom-right identity
            ], dim=1)                                                      # Shape: [kMax, n]
            
            T = torch.cat([T, Tinv], dim=0)                                # Final shape: [2*kMax, n]

        return T

    def _toFourierSpace(self, x):
        """ 
        x[nBatch, dv, nX] -> [nBatch, dv, fX = nX/2+1]
        x[nBatch, dv, nX, nY] -> [nBatch, dv, fX = nX, fY = nY/2+1]
        x[nBatch, dv, nX, nY, nZ] -> [nBatch, dv, fX = nX, fY = nY, fZ = nZ/2+1]
        """
        if x.dtype == torch.float16:
            x = x.float()  # cast to float32 for FFT

        if self.dim == 1:
            x = torch.fft.rfft(x, dim=-1, norm="ortho")
        elif self.dim == 2:
            x = torch.fft.rfftn(x, dim=(-2,-1), norm="ortho")      # RFFT on last 2 dimensions
        else:
            x = torch.fft.rfftn(x, dim=(-3,-2,-1), norm="ortho")   # RFFT on last 3 dimensions
        return x

    def _toRealSpace(self, x, org_size):
        """ 
        x[nBatch, dv, fX = nX/2+1] -> [nBatch, dv, nX]
        x[nBatch, dv, fX = nX, fY = nY/2+1] -> [nBatch, dv, nX, nY]
        x[nBatch, dv, fX = nX, fY = nY, fZ = nZ/2+1] -> [nBatch, dv, nX, nY, nZ]
        """
        if x.dtype == torch.complex32:
            x = x.cfloat()  # cast to complex64 for IFFT

        if self.dim == 1:
            x = torch.fft.irfft(x, n=org_size[-1], dim=-1, norm="ortho")
        elif self.dim == 2:
            x = torch.fft.irfftn(x, s=org_size, dim=(-2,-1), norm="ortho")     # IRFFT on last 2 dimensions
        else:
            x = torch.fft.irfftn(x, s=org_size, dim=(-3,-2,-1), norm="ortho")  # IRFFT on last 3 dimensions
        return x


    def forward(self, x:torch.tensor):
        """ x[nBatch, dv, nX, nY, ..] -> [nBatch, dv, nX, nY, ..] """
        org_size = x.shape[-self.dim:]

        # Transform to Fourier space -> [nBatch, dv, fX, fY,..]
        x = self._toFourierSpace(x)  # complex64
        f_dims = x.shape[-self.dim:]

        if self.training and self.use_complex_amp:
            einsum_fn = einsum_complexhalf
            R = deformat_complexTensor(self.R.half()).to(x.device)  # complex32
            x = torch.complex(x.real.half(), x.imag.half())         # complex32
        else:
            einsum_fn = torch.einsum
            R = deformat_complexTensor(self.R).to(x.device)     # complex64
            
        if self.dim == 1:
            Tx = self.T(self.kX, f_dims[0], x.device, sym=False)
            # -- Tx[kX, fX]
            x = einsum_fn("ax,eix->eia", Tx, x)   

            # Apply R[dv, dv, kX] -> [nBatch, dv, kX]
            x = einsum_fn("ija,eja->eia", R, x)   
            
            # Padding on high frequency modes -> [nBatch, dv, fX]
            x = einsum_fn("xa,eia->eix", Tx.T, x)      

        elif self.dim == 2:
            Tx = self.T(self.kX, f_dims[0], x.device, sym=True)
            Ty = self.T(self.kY, f_dims[1], x.device)
            # Truncate and keep only first modes -> [nBatch, dv, kX, kY,..]
            # -- Tx[kX, fX], Ty[kY, fY]
            x = einsum_fn("ax,by,eixy->eiab", Tx, Ty, x)

            # Apply R[dv, dv, kX, kY] -> [nBatch, dv, kX, kY]
            x = einsum_fn("ijab,ejab->eiab", R, x)

            # Padding on high frequency modes -> [nBatch, dv, fX, fY]
            x = einsum_fn("xa,yb,eiab->eixy", Tx.T, Ty.T, x)

        else:
            Tx = self.T(self.kX, f_dims[0], x.device, sym=True)
            Ty = self.T(self.kY, f_dims[1], x.device, sym=True)
            Tz = self.T(self.kZ, f_dims[2], x.device)
            # -- Tx[kX, fX], Ty[kY, fY], Tz[kZ, fZ]
            x = einsum_fn("ax,by,cz,ejxyz->ejabc", Tx, Ty, Tz, x)

            #  Apply R[dv, dv, kX, kY, kZ] -> [nBatch, dv, kX, kY, kZ]
            x = einsum_fn("ijabc,ejabc->eiabc", R, x)

           # Padding on high frequency modes -> [nBatch, dv, fX, fY, fZ]
            x = einsum_fn("xa,yb,zc,eiabc->eixyz", Tx.T, Ty.T, Tz.T, x)

        # Transform back to Real space -> [nBatch, dv, nX, nY, ..]
        # Need to pass signal orginal shape to round irfftn() 
        # if last dim is odd
        x = self._toRealSpace(x, org_size)
        x = x.half() if self.training and self.use_complex_amp else x   # float16

        if self.bias is not None:
            x = x + self.bias

        return x
