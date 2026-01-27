import torch
class VandermondeTransform:
    """
    Class for 1,2-dimensional Fourier transforms on a nonequispaced lattice of data
    ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py 
    """
    def __init__(self, positions, kX, kY=None, dim=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.kX = kX
        self.kY = kY if kY is not None else None
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        positions -= torch.min(positions)
        self.positions = positions * 6.28 / (torch.max(positions))
        self.batch_size = self.positions.shape[0]

        self.Vt, self.Vc = self.make_1Dmatrix()

        
    def make_1Dmatrix(self):
  
        with torch.no_grad():
            forward_mat = torch.zeros([self.batch_size, self.kX, self.positions.shape[1]], dtype=torch.cfloat, device=self.device)
            for row in range(self.kX):
                forward_mat[:, row, :] = torch.exp(-1j * (row - int(self.kX/2))* self.positions[:, :])

            inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)

        return forward_mat, inverse_mat
    
    
    def forward(self, data):
        """
        data: [batchsize, nParticle, dv]
        returns: [batchsize, modes, dv]
        """

        if data.device != self.device:
            data = data.to(self.device)
    
        return torch.bmm(self.Vt, data)  
        
    def inverse(self, data):
        """
        data: [batchsize, modes, dv,]
        returns: [batchsize, nParticle, dv]
        """
      
        return torch.bmm(self.Vc, data) 
        

