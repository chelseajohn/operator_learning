import torch
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
            self.Vt, self.Vc = self.make_1Dmatrix()
        else:
            self.kY = kY if kY is not None else kX
            y_positions = y_positions - torch.min(y_positions)
            self.y_positions = y_positions * 6.28 / torch.max(y_positions)
            self.Y_ = torch.cat((torch.arange(self.kY,dtype=dtype, device=device),
                                 torch.arange(start=-(self.kY), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None] # [B, 2kY, 1]
            self.Vt, self.Vc = self.make_2Dmatrix()
            
    def make_1Dmatrix(self):
  
        with torch.no_grad():
            m = self.kX*2
            xpos = self.x_positions.to(device=self.device, dtype=self.X_.dtype)
            X = torch.bmm(self.X_, xpos[:, None, :])   # [B, 2kX, N]

            # flatten to [B, m, N]
            forward_mat = torch.exp(-1j * X)

            inverse_mat = torch.conj(forward_mat).permute(0,2,1)

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

            # flatten to [B, m, N]
            forward_mat = torch.exp(-1j * phase).reshape(self.batch_size, m, self.number_points)
            inverse_mat = torch.conj(forward_mat).permute(0,2,1)
            #X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, self.kY*2, 1).to(self.device)
            #Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.kX*2).reshape(self.batch_size,m,self.number_points)).to(self.device)
            #forward_mat = torch.exp(-1j* (X_mat+Y_mat)).to(dtype=torch.cfloat, device=self.device) # [batchsize, m, nParticles]

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
        

