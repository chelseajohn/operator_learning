import torch
from operator_learning.utils.misc import einsum_complexhalf

class VandermondeTransformMatrixFree:
    """
    Matrix-free 1D/2D/3D Fourier transforms on a nonequispaced lattice.
    """
    def __init__(self, x_positions, kX, x_pos_min=None, x_pos_max=None, 
                 y_positions=None, kY=None, y_pos_min=None, y_pos_max=None,
                 z_positions=None, kZ=None, z_pos_min=None, z_pos_max=None,
                 dim=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        assert dim in (1, 2, 3), "dim must be 1 or 2 or 3"
        self.dim = dim
        self.kX = kX
        if x_pos_min is None:
            x_pos_min = torch.min(x_positions) 
        if x_pos_max is None:
            x_pos_max = torch.max(x_positions)
        x_positions = x_positions - x_pos_min              
        self.x_positions = x_positions * 2*torch.pi /  x_pos_max 
        self.X_ = torch.cat((torch.arange(self.kX, dtype=dtype, device=device), 
                             torch.arange(start=-(self.kX), end=0, dtype=dtype, device=device)), 
                             0)
        self.batch_size = x_positions.shape[0]
        self.number_points = x_positions.shape[1]
        self.Fx = torch.exp(-1j * self.X_[None, :, None] * self.x_positions[:, None, :]) #(batchsize, 2*kX, nParticle)

        if dim > 1:  
            if y_pos_min is None:
                y_pos_min = torch.min(y_positions) 
            if y_pos_max is None:
                y_pos_max = torch.max(y_positions)
            self.kY = kY if kY is not None else kX
            self.Y_ = torch.cat((torch.arange(self.kY, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0)
            y_positions = y_positions - y_pos_min              
            self.y_positions = y_positions * 2*torch.pi /y_pos_max  
            self.Fy = torch.exp(-1j * self.Y_[None, :, None] * self.y_positions[:, None, :]) #(batchsize, 2*kY, nParticle)
        if dim > 2:
            if z_pos_min is None:
                z_pos_min = torch.min(z_positions) 
            if z_pos_max is None:
                z_pos_max = torch.max(z_positions)
            self.kZ = kZ if kZ is not None else kX
            self.Z_ = torch.cat((torch.arange(self.kZ, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kZ), end=0, dtype=dtype, device=device)),
                                 0)
            z_positions = z_positions - z_pos_min                          
            self.z_positions = z_positions * 2*torch.pi / z_pos_max             
            self.Fz = torch.exp(-1j * self.Z_[None, :, None] * self.z_positions[:, None, :]) #(batchsize, 2*kZ, nParticle)


    def _forward_1d(self, data):
        """
        data: [batchsize, dv, nParticle]  
        out:  [batchsize, dv, 2*kX] 
        """
        
        return torch.einsum("bcp,bkp->bck", data, self.Fx)
      

    def _inverse_1d(self, data):
        """
        data: [batchsize, dv, 2*kX]
        out:  [batchsize, dv, nParticle]
        """
     
        return torch.einsum("bck,bkp->bcp", data, torch.conj(self.Fx))


    def _forward_2d(self, data):
        """
        data: [batchsize, dv, nParticle]
        out:  [batchsize, dv, (2*kX)*(2*kY)]
        """
    
        out = torch.einsum("bcp,bkp,blp->bckl", data, self.Fx, self.Fy)
        out = out.reshape(self.batch_size, data.shape[1], len(self.X_)*len(self.Y_))
        # print(f'data: {data.dtype}, Fx: {self.Fx.dtype}, Fy: {self.Fy.dtype}, FWD out: {out.dtype}', flush=True)
       
        return out 
    
    def _inverse_2d(self, data):
        """
        data: [batchsize, dv, (2*kX)*(2*kY)]
        out:  [batchsize, dv, nParticle]
        """
        d = data.reshape(self.batch_size, data.shape[1], len(self.X_), len(self.Y_))
        out = torch.einsum("bckl,bkp,blp->bcp", d, torch.conj(self.Fx), torch.conj(self.Fy)) 
        # print(f'BWD out: {out.dtype}, data: {data.dtype}, Fx: {self.Fx.dtype}, Fy: {self.Fy.dtype}', flush=True)

        return out
    
    def _forward_3d(self, data):
        """
        data: [batchsize, dv, nParticle]
        out:  [batchsize, dv, (2*kX)*(2*kY)*(2*kZ)]
        """
    
        out = torch.einsum("bcp,bkp,blp,bmp->bcklm", data, self.Fx, self.Fy, self.Fz)
        out = out.reshape(self.batch_size, data.shape[1], len(self.X_)*len(self.Y_)*len(self.Z_))

        return out 
    
    def _inverse_3d(self, data):
        """
        data: [batchsize, dv, (2*kX)*(2*kY)*(2*kZ)]
        out:  [batchsize, dv, nParticle]
        """
        d = data.reshape(self.batch_size, data.shape[1], len(self.X_), len(self.Y_), len(self.Z_))
        out = torch.einsum("bcklm,bkp,blp,bmp->bcp", d, torch.conj(self.Fx), torch.conj(self.Fy), torch.conj(self.Fz)) 

        return out
        
    def forward(self, data):
        """
        data: [batchsize, dv, nParticle]
        returns: [batchsize, dv, modes]
        """
        if data.device.type != self.device:
            data = data.to(self.device)

        if self.dim == 1:
            return self._forward_1d(data)
        elif self.dim == 2:
            return self._forward_2d(data)
        else:
            return self._forward_3d(data)

    def inverse(self, data):
        """
        data: [batchsize, dv, modes]
        returns: [batchsize, dv, nParticle]
        """
        if self.dim == 1:
            return self._inverse_1d(data)
        elif self.dim == 2:
            return self._inverse_2d(data)
        else:
            return self._inverse_3d(data)