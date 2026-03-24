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
    (ndim, klength). 
    """

    def __init__(self, device,  dataClass='pic', transform='kb', dim=1, dtype=torch.float32):
    
        assert dim == 1, "KbNUFFT can transform only image data"
        self.device = device
        self.dim = dim
        self.transform = transform
        self.dtype = dtype


    def build_ktraj_from_particles(self, np):
        """
        Input:
           np: number of modes which is same as number of particles
        Returns:
           k : fourier modes
        """

        with torch.no_grad():
            # NUFFT modules
            self.im_size = (np,)
            self.grid_size = (2*np, ) # internal FFT grid size >= np
            self.kbnufft = KbNufft(im_size=self.im_size, grid_size=self.grid_size).to(device=self.device)
            self.adjkb = KbNufftAdjoint(im_size=self.im_size, grid_size=self.grid_size).to(device=self.device)
            # self.toep = ToepNufft().to(device=self.device)

            k = torch.arange(-np//2, np//2, dtype=self.dtype)
      
        return k

    def forward(self, data):
        """
        Forward: data -> non-uniform k-space samples using KbNufft.
        data shape: (batchsize, dv, nParticle)
        returns kspace: (batchsize, dv, nParticle)
        """
 
        if data.device != self.device:
            data = data.to(self.device)

        b, c, p = data.shape
        modes = self.build_ktraj_from_particles(p)
        self.ktraj = modes[None, None, :].repeat(b*c, 1,1)  # [batchsize*dv, 1, nParticle]

        data_fwd = self.kbnufft(data.reshape(b*c, 1, p), self.ktraj, norm='ortho') # [batchsize*dv, 1, nParticle]

        # if self.transform == 'toeplitz':
        #     kernel = calc_toeplitz_kernel(omega=self.ktraj, im_size=self.im_size, 
        #                                   grid_size=self.grid_size, norm='ortho').unsqueeze(1) # [batchsize*dv, 1,nParticle]
        #     self.data_inv = self.toep(data, kernel) # [batchsize*dv, 1, nParticle]

        return data_fwd.reshape(b, c, p)

    def inverse(self, data):
        """
        Performing adjoint with KB 
        data shape: (batchsize, dv, nParticle)
        returns data: (batchsize, dv, nParticle)
        """

        b, c, p = data.shape
        if self.transform == 'kb':
            data_inv = self.adjkb(data.reshape(b*c, 1, p), self.ktraj, norm='ortho') # [batchsize*dv, 1, nParticle]
          
        return data_inv.reshape(b,c,p)

  