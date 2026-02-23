import torch
import numpy as np
from torchkbnufft import KbNufft, KbNufftAdjoint, ToepNufft, calc_toeplitz_kernel

class NUFFTTransform:
    """
    Toeplitz and Kaiser-Bessel NUFFT wrapper for 1D/2D usage using torchkbnufft.
    This class provides:
      - forward(image) -> Channels samples at Spectral Modes using KbNufft
      - adjoint(kspace) -> image (AdjKbNufft)
      - toep(image) -> applies T ≈ A' A (ToepNufft) using precomputed kernel

    Note: torchkbnufft expects `omega` / `ktraj` in radians per voxel(grid unit) and shape
    (ndim, klength). Construct ktraj from `x_data`, `y_data` (particle positions)
    by mapping particle positions into frequency coordinates in radians/voxel.
    """

    def __init__(self, device,  dataClass='pic', transform='toeplitz', dv=64, kX=None, kY=None, dim=1, dtype=torch.float32):
    
        assert dim in (1, 2)
        self.device = device
        self.dim = dim
        self.kX = kX
        self.kY = kY if kY is not None else None
        self.dv = dv
        self.transform = transform
        self.dtype = dtype
        if transform == 'toeplitz':
            assert self.dim == 1, "Toeplitz transform only supports dim=1"

        

    def build_ktraj_from_particles(self, x_data, y_data=None):
        """
        Input:
            x_data: [B, nParticles]
            y_data: [B, nParticles] (required for 2D)

        Output:
            1D: ktraj = [B, 1, nParticles]
            2D: ktraj = [B, 2, nParticles]
        """

        with torch.no_grad():
            # NUFFT modules
            self.im_size = (x_data.shape[1],)
            self.kbnufft = KbNufft(im_size=self.im_size).to(device=self.device)
            self.adjkb = KbNufftAdjoint(im_size=self.im_size).to(device=self.device)
            self.toep = ToepNufft().to(device=self.device)
            xPos = x_data.real
            x_min = xPos.min(dim=0, keepdim=True).values
            x_max = xPos.max(dim=0, keepdim=True).values

            if torch.any((x_max - x_min) == 0):
                raise ValueError("Some batches in xPos have zero range.")

            # normalize per batch: [0,1] and map to [-pi, pi]
            x_norm = (xPos - x_min) / (x_max - x_min)
            omega_x = ((x_norm - 0.5) * (2.0 * np.pi)).to(self.dtype)

            if self.dim == 1:
                # shape: [B, 1, particle]
                omega = omega_x.unsqueeze(1)
            else:
                assert y_data is not None, "y_data required for 2D"
                yPos = y_data.real
                # normalize per batch: [0,1] and map to [-pi, pi]
                y_min = yPos.min(dim=0, keepdim=True).values
                y_max = yPos.max(dim=0, keepdim=True).values

                if  torch.any((y_max - y_min) == 0):
                    raise ValueError("Some batches have zero range in  y_data.")

                y_norm = (yPos - y_min) / (y_max - y_min)
                omega_y = ((y_norm - 0.5) * (2.0 * np.pi)).to(self.dtype)

                # shape: [B, 2, particle]
                omega = torch.stack([omega_x, omega_y], dim=1)
            
        return omega.to(self.device) 

    def forward(self, data):
        """
        Forward: data -> non-uniform k-space samples using KbNufft.
        data shape: (B, C, nparticle)
        returns kspace: (B, C, nparticle)
        """
        with torch.no_grad():
            if data.device != self.device:
                data = data.to(self.device)

            if self.dim == 1:
                self.omega = self.build_ktraj_from_particles(x_data=data[:,0,:])
            else:
                self.omega = self.build_ktraj_from_particles(x_data=data[:,0,:], y_data=data[:,1,:])
            
            # toeplitz tranform for 2D not possible due to data layout
            if self.transform == 'toeplitz' and  self.dim == 1: 
                self.kernel = calc_toeplitz_kernel(self.omega, im_size=self.im_size)
                if len(self.kernel.size()) == 2:
                    self.kernel = self.kernel.unsqueeze(1)  # [B, 1, 2*nparicles]
                data_fwd = self.toep(data, self.kernel)/self.dv  # [B, C, nparticle]
            else:
                data_fwd = self.kbnufft(data, self.omega) # [B, C, nparticle]
        print(f'data: {data.shape}')
        print(f'data_fwd: {data_fwd.shape}')

        return data_fwd

    def inverse(self, data):
        """
        Performing adjoint then scaling to mimic IFFT
        data shape: (B, nparticle, Modes)
        returns data: (B, nparticle, Channel)
        """
        with torch.no_grad():
            if self.transform == 'toeplitz':
                data_inv = self.toep(data, self.kernel) # [B, C, nparticle]
            else:
                data_inv = self.adjkb(data, self.omega) # [B, C, nparticle]
        print(f'data: {data.shape}')
        print(f'data_inv: {data_inv.shape}')

        return data_inv/self.dv

  