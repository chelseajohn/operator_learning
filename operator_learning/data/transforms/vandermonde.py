import torch
from operator_learning.utils.misc import einsum_complexhalf
from operator_learning.utils.misc import print_rank0

class VandermondeTransform:
    """
    Class for 1,2,3-dimensional Fourier transforms on a nonequispaced lattice of data
    ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py 
    """
    def __init__(self, x_positions, kX, y_positions=None, kY=None,
                z_positions=None, kZ=None,dim=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        assert dim in (1, 2,3), "dim must be 1 or 2 or 3"
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
            self.Vt, self.Vc = self.make_1Dmatrix()
        elif dim == 2:
            self.kY = kY if kY is not None else kX
            y_positions = y_positions - torch.min(y_positions)
            self.y_positions = y_positions * 6.28 / torch.max(y_positions)
            self.Y_ = torch.cat((torch.arange(self.kY, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kY, 1]
            self.Vt, self.Vc = self.make_2Dmatrix()
        else:
            self.kY = kY if kY is not None else kX
            self.kZ = kZ if kZ is not None else kZ
            y_positions = y_positions - torch.min(y_positions)
            z_positions = z_positions - torch.min(z_positions)
            self.y_positions = y_positions * 6.28 / torch.max(y_positions)
            self.z_positions = z_positions * 6.28 / torch.max(z_positions)
            self.Y_ = torch.cat((torch.arange(self.kY, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kY, 1]
            self.Z_ = torch.cat((torch.arange(self.kZ, dtype=dtype, device=device),
                                 torch.arange(start=-(self.kZ), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kZ, 1]
            # print_rank0(f'Position shape: {x_positions.shape, y_positions.shape, z_positions.shape}')
            self.Vt, self.Vc = self.make_3Dmatrix()


    def make_1Dmatrix(self):
  
        with torch.no_grad():
            m = self.kX*2
            xpos = self.x_positions.to(device=self.device, dtype=self.X_.dtype)
            X = torch.bmm(self.X_, xpos[:, None, :])   # [B, 2kX, N]

            # flatten to [B, m, N]
            forward_mat = torch.exp(-1j * X)
            inverse_mat = torch.conj(forward_mat)

        return forward_mat, inverse_mat
              
            
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
            phase = phase.permute(0, 2, 1, 3)              # [B, Ky, Kx, N]

            # flatten to [B, m, N]
            forward_mat = torch.exp(-1j * phase).reshape(self.batch_size, m, self.number_points)
            inverse_mat = torch.conj(forward_mat)
            # X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, self.kY*2, 1).to(self.device)
            # Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.kX*2).reshape(self.batch_size,m,self.number_points)).to(self.device)
            # forward_mat = torch.exp(-1j* (X_mat+Y_mat)).to(dtype=torch.cfloat, device=self.device) # [batchsize, m, nParticles]

        return forward_mat, inverse_mat
    
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
            inverse_mat = torch.conj(forward_mat)

            return forward_mat, inverse_mat
    
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
            data_inv = einsum_complexhalf('bck,bkp->bcp', data, self.Vc)
        else:
            data_inv = torch.bmm(data, self.Vc) 

        return data_inv
        

