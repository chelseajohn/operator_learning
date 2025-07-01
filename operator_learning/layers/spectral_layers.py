import torch
import torch.nn as nn
from operator_learning.utils.misc import format_complexTensor, deformat_complexTensor

class SpectralConv(nn.Module):
    """
    Spectral convolution with FFT, linear transform, and Inverse FFT. 

    Args:
        dv (int): channels
        kX, kY, kZ (int): Fourier modes (Z only for 3D)
        bias (bool): bias for Fourier layer. Default is False.
        order (int): spatial dim (2 or 3). Default is 2.

  
    """
    def __init__(self, 
                 dv, 
                 kX, 
                 kY,
                 kZ=None,
                 bias=False,
                 order=2
    ):
        super().__init__()

        assert order in (2, 3), "Only 2D or 3D supported"
        self.order = order
        self.dv = dv
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        # k_max = 12 in http://arxiv.org/pdf/2010.08895
        self.kX = kX           
        self.kY = kY
        self.kZ = kZ 

        # R
        if order == 2:
            operator_weights = torch.rand(dv, dv, 2 * kX , kY, dtype=torch.cfloat)
            
        else:
            assert kZ is not None, "kZ must be specified for 3D"
            operator_weights = torch.rand(dv, dv, 2 * kX, 2 * kY, kZ, dtype=torch.cfloat)
        self.R = nn.Parameter(format_complexTensor(operator_weights))


        if bias:
            if order == 2:
                self.init_std = (2 / (dv + dv))**0.5
            else:
                self.init_std = (2/ (dv + dv + dv))**0.5
            self.bias = nn.Parameter(
                self.init_std * torch.randn(*(tuple([dv]) + (1,) * order))
            )
        else:
            self.init_std = None
            self.bias = None

    def T(self, kMax, n, device, sym=False):
        T = torch.cat([
            torch.eye(kMax, dtype=torch.cfloat),        # Top-left identity
            torch.zeros(kMax, n - kMax)                 # Zero-pad to match n columns
        ], dim=1)                                       # Shape: [kMax, n]

        if sym:
            Tinv = torch.cat([
                torch.zeros(kMax, n - kMax),            # Zero-pad on the left
                torch.eye(kMax, dtype=torch.cfloat)     # Bottom-right identity
            ], dim=1)                                   # Shape: [kMax, n]
            
            T = torch.cat([T, Tinv], dim=0)             # Final shape: [2*kMax, n]

        return T.to(device)

    def _toFourierSpace(self, x):
        """ 
        x[nBatch, dv, nX, nY] -> [nBatch, dv, fX = nX, fY = nY/2+1]
        x[nBatch, dv, nX, nY, nZ] -> [nBatch, dv, fX = nX, fY = nY, fZ = nZ/2+1]
        """
        if self.order == 2:
            x = torch.fft.rfft2(x, norm="ortho")   # RFFT on last 2 dimensions
        else:
            x = torch.fft.rfft3(x, norm="ortho")   # RFFT on last 3 dimensions
        return x

    def _toRealSpace(self, x, org_size):
        """ 
        x[nBatch, dv, fX = nX, fY = nY/2+1] -> [nBatch, dv, nX, nY]
        x[nBatch, dv, fX = nX, fY = nY, fZ = nZ/2+1] -> [nBatch, dv, nX, nY, nZ]
        """
        if self.order == 2:
            x = torch.fft.irfft2(x, s=org_size, norm="ortho")  # IRFFT on last 2 dimensions
        else:
            x = torch.fft.irfft3(x, s=org_size, norm="ortho")  # IRFFT on last 3 dimensions
        return x


    def forward(self, x:torch.tensor):
        """ x[nBatch, dv, nX, nY, ..] -> [nBatch, dv, nX, nY, ..] """
        org_size = x.shape[-self.order:]
        # Transform to Fourier space -> [nBatch, dv, fX, fY,..]
        x = self._toFourierSpace(x)
        # Truncate and keep only first modes -> [nBatch, dv, kX, kY,..]
        f_dims = x.shape[-self.order:]

        Tx = self.T(self.kX, f_dims[0], x.device, sym=True)

        if self.order == 2:
            Ty = self.T(self.kY, f_dims[1], x.device)
            # -- Tx[kX, fX], Ty[kY, fY]
            x = torch.einsum("ax,by,eixy->eiab", Tx, Ty, x)

            # Apply R[dv, dv, kX, kY] -> [nBatch, dv, kX, kY]
            R = deformat_complexTensor(self.R).to(x.device)
            x = torch.einsum("ijab,ejab->eiab", R, x)

            # Padding on high frequency modes -> [nBatch, dv, fX, fY]
            x = torch.einsum("xa,yb,eiab->eixy", Tx.T, Ty.T, x)

        else:
            Ty = self.T(self.kY, f_dims[1], x.device, sym = True)
            Tz = self.T(self.kZ, f_dims[2], x.device)
            # -- Tx[kX, fX], Ty[kY, fY], Tz[kZ, fZ]
            x = torch.einsum("ax,by,cz,ejxyz->eijabc", Tx, Ty, Tz, x)

            #  Apply R[dv, dv, kX, kY, kZ] -> [nBatch, dv, kX, kY, kZ]
            R = deformat_complexTensor(self.R).to(x.device)
            x = torch.einsum("ijabc,ejabc->eiabc", R, x)

           # Padding on high frequency modes -> [nBatch, dv, fX, fY, fZ]
            x_padded = torch.einsum("xa,yb,zc,eiabc->eixyz", Tx.T, Ty.T, Tz.T, x)


        # Transform back to Real space -> [nBatch, dv, nX, nY, ..]
        # Need to pass signal orginal shape to round irfftn() 
        # if last dim is odd
        x = self._toRealSpace(x, org_size)

        if self.bias is not None:
            x = x + self.bias

        return x
