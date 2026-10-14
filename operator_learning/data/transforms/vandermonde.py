import torch
from operator_learning.utils.misc import einsum_complexhalf
from operator_learning.utils.misc import print_rank0

class VandermondeTransform:
    """
    Class for 1,2,3-dimensional Fourier transforms on a nonequispaced lattice of data
    ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py 
    Particle positions are assumed to be normalized to [0, 1] by default and
    are scaled to [0, 2π) for the Fourier transform. Custom spatial bounds can
    be provided through the min/max position arguments
    """
    def __init__(self, x_positions, kX, x_pos_min=0.0, x_pos_max=1.0, 
                 y_positions=None, kY=None, y_pos_min=0.0, y_pos_max=1.0,
                 z_positions=None, kZ=None, z_pos_min=0.0, z_pos_max=1.0,
                dim=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        assert dim in (1, 2,3), "dim must be 1 or 2 or 3"
        self.dim = dim
        self.kX = kX
        if x_pos_min != 0.0 or x_pos_max != 1.0:
            x_positions = (x_positions - x_pos_min) / x_pos_max             
        self.x_positions = x_positions*2*torch.pi 
        self.batch_size = x_positions.shape[0]
        self.number_points = x_positions.shape[1]
        self.X_ = torch.cat((torch.arange(self.kX, dtype=dtype, device=device), 
                             torch.arange(start=-(self.kX), end=0, dtype=dtype, device=device)), 
                             0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kX, 1]
        
        
        if dim == 1:
            self.Vt = self.make_1Dmatrix()
        elif dim == 2:
            self.kY = kY if kY is not None else kX
            self.Y_ = torch.cat((torch.arange(self.kY, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kY, 1]
            if y_pos_min != 0.0 or y_pos_max != 1.0:
                y_positions = (y_positions - y_pos_min) / y_pos_max              
            self.y_positions = y_positions*2*torch.pi 
            self.Vt = self.make_2Dmatrix()
        else:
            self.kY = kY if kY is not None else kX
            self.Y_ = torch.cat((torch.arange(self.kY, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kY, 1]
            if y_pos_min != 0.0 or y_pos_max != 1.0:
                y_positions = (y_positions - y_pos_min) / y_pos_max              
            self.y_positions = y_positions*2*torch.pi 

            self.kZ = kZ if kZ is not None else kZ
            self.Z_ = torch.cat((torch.arange(self.kZ, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kZ), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kZ, 1]
            if z_pos_min != 0.0 or z_pos_max != 1.0:
                z_positions = (z_positions - z_pos_min) / z_pos_max                  
            self.z_positions = z_positions*2*torch.pi   
            self.Vt = self.make_3Dmatrix()

    def make_1Dmatrix(self):
  
        with torch.no_grad():
            m = self.kX*2
            xpos = self.x_positions.to(device=self.device, dtype=self.X_.dtype)
            X = torch.bmm(self.X_, xpos[:, None, :])   # [B, 2kX, N]

            # flatten to [B, m, N]
            forward_mat = torch.exp(-1j * X)
        
        return forward_mat
              
            
    def make_2Dmatrix(self):
        
        with torch.no_grad():
            m = (self.kX*2)*(self.kY*2)
            xpos = self.x_positions.to(device=self.device, dtype=self.X_.dtype) # [B, N]
            ypos = self.y_positions.to(device=self.device, dtype=self.Y_.dtype) # [B, N]
            X = torch.bmm(self.X_, xpos[:, None, :])   # [B, 2kX, N]
            Y = torch.bmm(self.Y_, ypos[:, None, :])   # [B, 2kY, N]

            # make grid: [B, 2kX, 2kY, N]
            phase = X[:, :, None, :] + Y[:, None, :, :]
            # The following permutation is only needed for using old model weights which were trained with that
            # convention. If we are training a new model from scratch then this is not needed. 
            # phase = phase.permute(0, 2, 1, 3)              # [B, 2Ky, 2Kx, N]

            # flatten to [B, m, N]
            forward_mat = torch.exp(-1j * phase).reshape(self.batch_size, m, self.number_points)
            

        return forward_mat
    
    def make_3Dmatrix(self):
        
        with torch.no_grad():
            m = (self.kX*2)*(self.kY*2)*(self.kZ*2)
            xpos = self.x_positions.to(device=self.device, dtype=self.X_.dtype) # [B, N]
            ypos = self.y_positions.to(device=self.device, dtype=self.Y_.dtype) # [B, N]
            zpos = self.z_positions.to(device=self.device, dtype=self.Z_.dtype) # [B, N]
            X = torch.bmm(self.X_, xpos[:, None, :])   # [B, 2kX, N]
            Y = torch.bmm(self.Y_, ypos[:, None, :])   # [B, 2kY, N]
            Z = torch.bmm(self.Z_, zpos[:, None, :])   # [B, 2kZ, N]

            # make grid: [B, 2kX, 2kY, 2kZ, N]
            phase = X[:, :, None, None, :] + Y[:, None, :, None, :] + Z[:, None, None, :, :]

            # flatten to [B, m, N]
            forward_mat = torch.exp(-1j * phase).reshape(self.batch_size, m, self.number_points)

            return forward_mat
    
    def forward(self, data):
        """
        data: [batchsize, dv, nParticle]
        returns: [batchsize, dv, modes]
        """

        if data.device != self.device:
            data = data.to(self.device)
              
        # torch.bmm does not support complexHalf
        # 1D: [batchsize, dv, nParticle] x [batchsize, nParticle, kX]
        # 2D/3D: [batchsize, dv, nParticle] x [batchsize, nParticle, modes]
        if data.dtype == torch.complex32:
            data_fwd = einsum_complexhalf('bcp,bpk->bck', data, self.Vt.permute(0,2,1))
        else:
            data_fwd = torch.bmm(data, self.Vt.permute(0,2,1))  

        return data_fwd
        
    def inverse(self, data):
        """
        data: [batchsize, dv, modes]
        returns: [batchsize, dv, nParticle]
        """

        # torch.bmm does not support complexHalf
        # 1D: [batchsize, dv, kX] x [batchsize, kX, nParticle]
        # 2D/3D: [batchsize, dv, modes] x [batchsize, modes, nParticle]
        if data.dtype == torch.complex32:
            data_inv = einsum_complexhalf('bck,bkp->bcp', data, self.Vt.conj())
        else:
            data_inv = torch.bmm(data, self.Vt.conj()) 

        return data_inv
        

