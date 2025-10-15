import torch
import numpy as np

class VandermondeTransform:
    """
    Class for 1,2-dimensional Fourier transforms on a nonequispaced lattice of data
    ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py 
    """
    def __init__(self, device, kX, kY=None, dataset=None, dataClass='pic', dim=1, dtype=torch.complex64):
        self.device = device
        self.dtype = dtype
        self.kX = kX
        self.kY = kY if kY is not None else None
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        self.dataset = dataset
        self.dataClass = dataClass

        if dim == 1:
            # [1, kX, 1]
            self._X = torch.arange(self.kX, dtype=torch.float32, device=device)[None, :, None]
        else:
            # [1, 2*kX, 1]
            self._X = torch.cat((
                torch.arange(self.kX, dtype=torch.float32, device=device),
                torch.arange(start=-self.kX, end=0, dtype=torch.float32, device=device)
            ), dim=0)[None, :, None]

            # [1, 2*kY-1, 1]
            self._Y = torch.cat((
                torch.arange(self.kY, dtype=torch.float32, device=device),
                torch.arange(start=-(self.kY-1), end=0, dtype=torch.float32, device=device)
            ), dim=0)[None, :, None]

    
    def make_1Dmatrix(self, x_data):
        """Generates Vandermonde 1D matrices for forward and inverse transforms."""
        # x_data: [B, particle], V: [B, kX, particle]
        nParticle = x_data.shape[-1]

        if self.dataClass == 'rbc':
            xPos = torch.tensor(self.dataset.grid[0], dtype=torch.float32, device=self.device).repeat(x_data.shape[0], 1)
        else:
            xPos = x_data.real.to(self.device)

        # scaling btw 0 and 2*pi
        xPos = ((xPos - xPos.min()) / xPos.max()) * 2 * np.pi
        V = torch.exp(-1j * ((self._X - int(self.kX/2))* xPos[:, None, :])).to(self.dtype) # [1, kX, 1] x [B, 1, nX]
        
        norm = torch.sqrt(torch.tensor(nParticle, dtype=torch.int, device=self.device))
        self.V = (V / norm).to(self.device)

        return  self.V.clone().permute(0,2,1)
    
    def make_2Dmatrix(self, x_data, y_data):
        """Generates Vandermonde 2D matrices for forward and inverse transforms."""
        # x_data: [B, particle], y_data: [B, particle] 
        if self.dataClass == 'rbc':
            xPos = torch.tensor(self.dataset.grid[0], dtype=torch.float32, device=self.device).repeat(x_data.shape[0], 1)
            yPos = torch.tensor(self.dataset.grid[1], dtype=torch.float32, device=self.device).repeat(y_data.shape[0], 1)
        else:
            xPos = x_data.real.to(self.device)
            yPos = y_data.real.to(self.device)

        # scaling btw 0 and 2*pi
        xPos = ((xPos - xPos.min()) / xPos.max()) * 2 * np.pi
        yPos = ((yPos - yPos.min()) / yPos.max()) * 2 * np.pi

        # ToDO: implement for RBC
        m = (self.kX*2)*(self.kY*2-1)
        X_mat = torch.matmul(self._X, xPos[:,None,:]).repeat(1, (self.kY*2-1), 1)
        Y_mat = torch.matmul(self._Y, yPos[:,None,:]).repeat(1, 1, self.kX*2).reshape(yPos.shape[0], m, yPos.shape[-1]) # [B, m, particle]

        self.forward_mat = ((torch.exp(-1j* (X_mat+Y_mat))).to(self.dtype)/ xPos.shape[-1]).to(self.device)
       
        return self.forward_mat.clone().permute(0,2,1)

    def forward(self, data):

        """Computes the forward DSE transform."""

        if self.dim == 1:
           Vt = self.make_1Dmatrix(x_data=data[:, 0, :]) # [B, kX, particle]
           data_fwd = torch.bmm(data, Vt.to(data.device)) # [B, C, particle] x [B, particle, kX]
        else:
            # ToDO: implement for RBC
            Vt = self.make_2Dmatrix(x_data=data[:,0,:], y_data=data[:,1,:])  # [B, particle, m]
            data_fwd = torch.bmm(data, Vt.to(data.device)) # [B, C, particle] x [B, particle, m]
           
        return data_fwd
    
    def inverse(self, data):
        """Computes the inverse Fourier transform."""
        if self.dim == 1:
            # data: [B, C, kX], Vc: [B, kX, particle]
            Vc =  torch.conj(self.V.clone())
            data_inv = torch.matmul(data, Vc.to(data.device))
        else:
            # ToDO: implement for RBC
            #  data: [B, C, m], Vc: [B, m, particle] 
            Vc = torch.conj(self.forward_mat.clone()) 
            data_inv = torch.matmul(data, Vc.to(data.device)) 
   
        return data_inv

