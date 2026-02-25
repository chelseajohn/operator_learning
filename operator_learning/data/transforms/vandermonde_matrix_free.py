import torch
from operator_learning.utils.misc import einsum_complexhalf

class VandermondeTransformMatrixFree:
    """
    Matrix-free 1D/2D Fourier transforms on a nonequispaced lattice.
    """
    def __init__(self, x_positions, kX, y_positions=None, kY=None, dim=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        self.kX = kX
        x_positions = x_positions - torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / torch.max(x_positions)
        self.X_ = torch.cat((torch.arange(self.kX, dtype=dtype, device=device), 
                             torch.arange(start=-(self.kX), end=0, dtype=dtype, device=device)), 
                             0)
        self.batch_size = x_positions.shape[0]
        self.number_points = x_positions.shape[1]
        self.Fx = torch.exp(-1j * self.X_[None, :, None] * self.x_positions[:, None, :]) #(batchsize, dv, nParticle)

        if dim == 2:  
            self.kY = kY if kY is not None else kX
            self.Y_ = torch.cat((torch.arange(self.kY,dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0)
            y_positions = y_positions - torch.min(y_positions)
            self.y_positions = y_positions * 6.28 / torch.max(y_positions)
            self.Fy = torch.exp(-1j * self.Y_[None, :, None] * self.y_positions[:, None, :]) #(batchsize, dv, nParticle)
                
    
    def _forward_1d(self, data):
        """
        data: [batchsize, dv, nParticle]  
        out:  [batchsize, dv, kX] 
        """
        
        return torch.einsum("bcp,bkp->bck", data, self.Fx)
      

    def _inverse_1d(self, data):
        """
        data: [batchsize, dv, kX]
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

        return out 
    
    def _inverse_2d(self, data):
        """
        data: [batchsize, dv, (2*kX)*(2*kY)]
        out:  [batchsize, dv, nParticle]
        """
        d = data.reshape(self.batch_size, data.shape[1], len(self.X_), len(self.Y_))
        out = torch.einsum("bckl,bkp,blp->bcp", d, torch.conj(self.Fx), torch.conj(self.Fy)) 

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
        else:
            return self._forward_2d(data)

    def inverse(self, data):
        """
        data: [batchsize, dv, modes]
        returns: [batchsize, dv, nParticle]
        """
        if self.dim == 1:
            return self._inverse_1d(data)
        else:
            return self._inverse_2d(data)