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

        if dim == 1:
            self.Vt, self.Vc = self.make_1Dmatrix()
        else:
            self.kY = kY if kY is not None else kX
            y_positions = y_positions - torch.min(y_positions)
            self.y_positions = y_positions * 6.28 / torch.max(y_positions)
            self.X_ = torch.cat((torch.arange(kX, dtype=dtype, device=device), 
                                 torch.arange(start=-(kX), end=0, dtype=dtype, device=device)), 
                                 0).repeat(self.batch_size, 1)[:,:,None]
            #self.Y_ = torch.cat((torch.arange(kY,dtype=dtype, device=device),
            #                     torch.arange(start=-(kY-1), end=0, dtype=dtype, device=device)),
            #                     0).repeat(self.batch_size, 1)[:,:,None]
            self.Y_ = torch.cat((torch.arange(kY,dtype=dtype, device=device),
                                 torch.arange(start=-(kY), end=0, dtype=dtype, device=device)),
                                 0).repeat(self.batch_size, 1)[:,:,None]
            self.Vt, self.Vc = self.make_2Dmatrix()
            
    def make_1Dmatrix(self):
  
        with torch.no_grad():
            forward_mat = torch.zeros([self.batch_size, self.kX, self.positions.shape[1]], dtype=torch.cfloat, device=self.device)
            for row in range(self.kX):
                forward_mat[:, row, :] = torch.exp(-1j * (row - int(self.kX/2))* self.positions[:, :])

            #inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)
            inverse_mat = torch.conj(forward_mat).permute(0,2,1)

        return forward_mat, inverse_mat
    
    def make_2Dmatrix(self):
        
        with torch.no_grad():
            #m = (self.kX*2)*(self.kY*2-1)
            m = (self.kX*2)*(self.kY*2)
            #X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.kY*2-1), 1).to(self.device)
            X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, self.kY*2, 1).to(self.device)
            Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.kX*2).reshape(self.batch_size,m,self.number_points)).to(self.device)
            
            forward_mat = torch.exp(-1j* (X_mat+Y_mat)).to(dtype=torch.cfloat, device=self.device) # [batchsize, m, nParticles]
            #inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)
            inverse_mat = torch.conj(forward_mat).permute(0,2,1)

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
        

