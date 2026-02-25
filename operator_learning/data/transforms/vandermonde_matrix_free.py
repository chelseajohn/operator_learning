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
        if dim == 2:  
            self.kY = kY if kY is not None else kX
            self.Y_ = torch.cat((torch.arange(self.kY,dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0)
            y_positions = y_positions - torch.min(y_positions)
            self.y_positions = y_positions * 6.28 / torch.max(y_positions)
                
    
    def _forward_1d(self, data):
        """
        data: [batchsize, dv, nParticle]  
        out:  [batchsize, dv, kX] 
        """
        
        batch_size, channels, Np = data.shape
        out = torch.zeros(batch_size, channels, len(self.X_), dtype=torch.cfloat, device=data.device)
        for b in range(batch_size):
            xp = self.x_positions[b]  # (Np)
            for i, k in enumerate(self.X_):
                phase = torch.exp(-1j * k * xp)
                out[b, :, i] = torch.sum(data[b] * phase, dim=1)  # sum over particles

        return out
      

    def _inverse_1d(self, data):
        """
        data: [batchsize, dv, kX]
        out:  [batchsize, dv, nParticle]
        """
     
        batch_size, channels, nK = data.shape
        Np = self.x_positions.shape[1]

        out = torch.zeros(batch_size, channels, Np,
                        dtype=torch.cfloat, device=data.device)

        for b in range(batch_size):
            xp = self.x_positions[b]   # (Np)
            for i, k in enumerate(self.X_):
                phase = torch.exp(+1j * k * xp)   
                out[b] += data[b, :, i][:, None] * phase[None, :]

        return out


    def _forward_2d(self, data):
        """
        data: [batchsize, dv, nParticle]
        out:  [batchsize, dv, (2*kX)*(2*kY)]
        """
        
        batch_size, channels, Np = data.shape
        out = torch.zeros(batch_size, channels, len(self.X_)*len(self.Y_),
                        dtype=torch.cfloat, device=data.device)

        for b in range(batch_size):
            x = self.x_positions[b]   # (Np)
            y = self.y_positions[b]   # (Np)
            i = 0
            for ky in self.Y_:
                for kx in self.X_:
                    phase = torch.exp(-1j * (kx * x + ky * y))   # (Np)
                    out[b, :, i] = torch.sum(data[b] * phase, dim=1)
                    i += 1
        
        return out

    def _inverse_2d(self, data):
        """
        data: [batchsize, dv, (2*kX)*(2*kY)]
        out:  [batchsize, dv, nParticle]
        """
        batch_size, channels, m = data.shape
        Np = self.x_positions.shape[1]

        out = torch.zeros(batch_size, channels, Np,
                        dtype=torch.cfloat, device=data.device)

        for b in range(batch_size):
            xp = self.x_positions[b]
            yp = self.y_positions[b]
            i = 0
            for ky in self.Y_:
                for kx in self.X_:
                    phase = torch.exp(1j * (kx * xp + ky * yp))
                    out[b] += data[b, :, i][:, None] * phase[None, :]
                    i += 1

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