import torch
import numpy as np
from torchkbnufft import KbNufft, KbNufftAdjoint, ToepNufft, calc_toeplitz_kernel
import pytorch_finufft as fin

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
        self.dataClass = dataClass


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

  
class Finufft:
    """ 
    Supports 1D/2D/3D spatial transforms over non-uniform 
    particle positions using pytorch finufft
    """
    def __init__(self, x_positions, kX, x_pos_min=None, x_pos_max=None,
                 y_positions=None, kY=None, y_pos_min=None, y_pos_max=None,
                 z_positions=None, kZ=None, z_pos_min=None, z_pos_max=None,
                 dim=1, device='cuda', dtype=torch.float32):
        
        self.device = device
        assert dim in (1, 2, 3), "dim must be 1 or 2 or 3"
        self.dim = dim
        self.kX = 2*kX
        if x_pos_min is None:
            x_pos_min = torch.min(x_positions) 
        if x_pos_max is None:
            x_pos_max = torch.max(x_positions)
        x_positions = x_positions - x_pos_min              
        self.x_positions = x_positions * 2*torch.pi /  x_pos_max 
        self.batch_size = x_positions.shape[0]
        self.number_points = x_positions.shape[1]
        self.dtype = dtype
      
        if dim > 1:  
            if y_pos_min is None:
                y_pos_min = torch.min(y_positions) 
            if y_pos_max is None:
                y_pos_max = torch.max(y_positions)
            self.kY = 2*kY if kY is not None else 2*kX
            y_positions = y_positions - y_pos_min              
            self.y_positions = y_positions * 2*torch.pi /y_pos_max  
           
        if dim > 2:
            if z_pos_min is None:
                z_pos_min = torch.min(z_positions) 
            if z_pos_max is None:
                z_pos_max = torch.max(z_positions)
            self.kZ = 2*kZ if kZ is not None else 2*kX
            z_positions = z_positions - z_pos_min                          
            self.z_positions = z_positions * 2*torch.pi / z_pos_max             
            

    def _get_pts(self, t):
        """Get spatial coordinate tuple for timestep t."""
        if self.dim == 1:
            return self.x_positions[t]
        elif self.dim == 2:
            return torch.stack((self.x_positions[t], self.y_positions[t]))
        else:
            return torch.stack((self.x_positions[t], self.y_positions[t], self.z_positions[t]))

    @property
    def _n_modes(self):
        if self.dim == 1:
            return (self.kX)
        elif self.dim == 2:
            return (self.kX, self.kY)
        else:
            return (self.kX, self.kY, self.kZ)

    def _forward_single(self, pts, data_t):
        """
        pts:    (dim, N)
        data_t: (dv, N) complex
        returns (dv, modes_flat) complex
        """
        # finufft type1 with batched sources: values (dv, N) -> output (dv, *n_modes)
        out = fin.functional.finufft_type1(pts, data_t, self._n_modes)
        return out.reshape(data_t.shape[0], -1)  # (dv, modes_flat)

    def _inverse_single(self, pts, data_t):
        """
        pts:    (dim, N)
        data_t: (dv, modes_flat) complex
        returns (dv, N) complex
        """
        grid = data_t.reshape(data_t.shape[0], *self._n_modes)  # (dv, *n_modes)
        out = fin.functional.finufft_type2(pts, grid,)
        return out  # (dv, N)

    def forward(self, data):
        """
        data: (batch, dv, nParticle) 
        returns: (batch, dv, modes_flat) 
            modes_flat = 2*kX for 1D
                       = 2*kX * 2*kY for 2D
                       = 2*kX * 2*kY * 2*kZ for 3D
        """
        if data.device.type != self.device:
            data = data.to(self.device)
        results = []
        for t in range(self.batch_size):
            pts = self._get_pts(t)
            results.append(self._forward_single(pts, data[t]))
        return torch.stack(results)  # (batch, dv, modes_flat)

    def inverse(self, data):
        """
        data: (batch, dv, modes_flat) complex
        returns: (batch, dv, nParticle) complex
        """
        results = []
        for t in range(self.batch_size):
            pts = self._get_pts(t)
            results.append(self._inverse_single(pts, data[t]))
        return torch.stack(results)  # (batch, dv, nParticle)