import torch
from operator_learning.utils.misc import einsum_complexhalf

class VandermondeTransform:
    """
    Class for 1,2-dimensional Fourier transforms on a nonequispaced lattice of data
    ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py 
    """
    def __init__(self, x_positions, kX, y_positions=None, kY=None, dim=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        self.kX = kX
        x_positions = x_positions - torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / torch.max(x_positions)
        self.batch_size = x_positions.shape[0]
        self.number_points = x_positions.shape[1]
        self.X_ = torch.cat((torch.arange(self.kX, dtype=dtype, device=device), 
                             torch.arange(start=-(self.kX), end=0, dtype=dtype, device=device)), 
                             0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kX, 1]

        if dim == 1:
            self.X_ = torch.arange(kX, dtype=dtype, device=device)[None, :, None] # [1, kX, 1]
        else:
            self.kY = kY if kY is not None else kX
            y_positions = y_positions - torch.min(y_positions)
            self.y_positions = y_positions * 6.28 / torch.max(y_positions)
            self.X_ = torch.cat((torch.arange(kX, dtype=dtype, device=device), \
                                 torch.arange(start=-(kX), end=0, dtype=dtype, device=device)), 
                                 0)[None,:,None]   # [1, 2*kX, 1]
            self.Y_ = torch.cat((torch.arange(kY,dtype=dtype, device=device), \
                                 torch.arange(start=-(kY-1), end=0, dtype=dtype, device=device)),
                                 0)[None,:,None]   # [1, 2*kY-1, 1]
            
    def make_1Dmatrix(self):
  
        with torch.no_grad():
            # [1, kX, 1] x [batchsize, 1, nParticle]
            self.forward_mat = torch.exp(-1j * ((self.X_ - int(self.kX/2))* self.x_positions[:, None, :])) 

        return self.forward_mat 

              
    
    def make_2Dmatrix(self):
        
        with torch.no_grad():
            # m = (self.kX*2)*(self.kY*2-1)
            X_mat = torch.matmul(self.X_, self.x_positions[:,None,:]).repeat(1, (2*self.kY-1), 1).to(self.device)
            Y_mat = torch.matmul(self.Y_, self.y_positions[:,None,:]).repeat(1, 2*self.kX, 1).to(self.device)
            
            forward_mat = torch.exp(-1j* (X_mat+Y_mat))/self.number_points
            self.forward_mat = forward_mat.to(dtype=torch.cfloat, device=self.device) # [batchsize, m, nParticles]

        return self.forward_mat 
    
    def forward(self, data):
        """
        data: [batchsize, dv, nParticle]
        returns: [batchsize, dv, modes]
        """

        if data.device != self.device:
            data = data.to(self.device)
              
              
        if self.dim == 1:
            V = self.make_1Dmatrix()  # [batchsize, kX, nParticle]
        else:
            V = self.make_2Dmatrix()
        
        # torch.bmm does not support complexHalf
        # 1D: [batchsize, dv, nParticle] x [batchsize, nParticle, kX]
        # 2D: [batchsize, dv, nParticle] x [batchsize, nParticle, m]
        if data.dtype == torch.complex32:
            data_fwd = einsum_complexhalf('bcp,bpk->bck', data, V.permute(0,2,1))
        else:
            data_fwd = torch.bmm(data, V.permute(0,2,1))  

        return data_fwd
        
    def inverse(self, data):
        """
        data: [batchsize, dv, modes]
        returns: [batchsize, dv, nParticle]
        """

        Vc = torch.conj(self.forward_mat) 
      
        # torch.bmm does not support complexHalf
        # 1D data: [batchsize, dv, kX] x [batchsize, kX, nParticle]
        # 2D data: [batchsize, dv, m] x [batchsize, m, nParticle]
        if data.dtype == torch.complex32:
            data_inv = einsum_complexhalf('bck,bkp->bcp', data, Vc)
        else:
            data_inv = torch.bmm(data, Vc) 

        return data_inv
        

