import torch
import numpy as np
from operator_learning.utils.misc import print_rank0, einsum_complexhalf

class VandermondeTransform:
    """
    Class for 1,2-dimensional Fourier transforms on a nonequispaced lattice of data
    ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py 
    """
    def __init__(self, device, kX, kY=None, dataset=None, dataClass='pic', dim=1, dtype=torch.float32):
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
            self._X = torch.arange(self.kX, dtype=self.dtype, device=device)[None, :, None]
        else:
            # [1, 2*kX, 1]
            self._X = torch.cat((
                torch.arange(self.kX, dtype=self.dtype, device=device),
                torch.arange(start=-self.kX, end=0, dtype=self.dtype, device=device)
            ), dim=0)[None, :, None]

            # [1, 2*kY-1, 1]
            self._Y = torch.cat((
                torch.arange(self.kY, dtype=self.dtype, device=device),
                torch.arange(start=-(self.kY-1), end=0, dtype=self.dtype, device=device)
            ), dim=0)[None, :, None]

    def make_1Dmatrix(self, x_data):
        """
        Generates Vandermonde 1D matrices for forward and inverse transforms.
        """
        with torch.no_grad():
            # x_data: [B, particle], V: [B, kX, particle]
            nParticle = x_data.shape[-1]

            if self.dataClass == 'rbc':
                xPos = torch.tensor(self.dataset.grid[0], dtype=self.dtype, device=self.device).repeat(x_data.shape[0], 1)
            else:
                xPos = x_data.real

            # scaling btw 0 and 2*pi
            xPos = ((xPos - xPos.min()) / xPos.max()) * 2 * np.pi

            forward_mat = torch.exp(-1j * ((self._X.to(xPos.dtype) - int(self.kX/2))* xPos[:, None, :])) # [1, kX, 1] x [B, 1, nX]
            
            norm = 1.0 / torch.sqrt(torch.tensor(nParticle, dtype=torch.int, device=self.device))
            forward_mat.mul_(norm)
            self.forward_mat = forward_mat.to(x_data.dtype)

        return self.forward_mat
    
    def make_2Dmatrix(self, x_data, y_data):
        """
        Generates Vandermonde 2D matrices for forward and inverse transforms.
        """
        with torch.no_grad():
            # x_data: [B, particle], y_data: [B, particle] 
            if self.dataClass == 'rbc':
                xPos = torch.tensor(self.dataset.grid[0], dtype=self.dtype, device=self.device).repeat(x_data.shape[0], 1)
                yPos = torch.tensor(self.dataset.grid[1], dtype=self.dtype, device=self.device).repeat(y_data.shape[0], 1)
            else:
                xPos = x_data.real
                yPos = y_data.real

            # scaling btw 0 and 2*pi
            xPos = ((xPos - xPos.min()) / xPos.max()) * 2 * np.pi
            yPos = ((yPos - yPos.min()) / yPos.max()) * 2 * np.pi

            m = (self.kX*2)*(self.kY*2-1)
            X_mat = torch.matmul(self._X.to(xPos.dtype), xPos[:,None,:]).repeat(1, (self.kY*2-1), 1)
            Y_mat = torch.matmul(self._Y.to(yPos.dtype), yPos[:,None,:]).repeat(1, 1, self.kX*2).reshape(yPos.shape[0], m, yPos.shape[-1]) 

            forward_mat = ((torch.exp(-1j* (X_mat+Y_mat))/ xPos.shape[-1])) # [B, m, particle]
            self.forward_mat = forward_mat.to(x_data.dtype)

        return self.forward_mat

    def forward(self, data):
        """Computes the forward DSE transform."""

        with torch.no_grad():
            if data.device != self.device:
                data = data.to(self.device)

            if self.dim == 1:
                V = self.make_1Dmatrix(x_data=data[:, 0, :]) # [B, kX, particle]
            else:
                V = self.make_2Dmatrix(x_data=data[:,0,:], y_data=data[:,1,:])  # [B, m, particle]
            
            # torch.bmm does not support complexHalf
            # 1D: [B, C, particle] x [B, particle, kX], 2D: [B, C, particle] x [B, particle, m]
            if data.dtype == torch.complex32:
                data_fwd = einsum_complexhalf('bcp,bpk->bck', data, V.permute(0,2,1))
            else:
                data_fwd = torch.bmm(data, V.permute(0,2,1))  
            
        return data_fwd
        
    def inverse(self, data):
        """Computes the inverse Fourier transform."""
        
        with torch.no_grad():
            if self.dim == 1:
                # Vc: [B, kX, particle]
                Vc =  torch.conj(self.forward_mat)
            else:
                # Vc: [B, m, particle] 
                Vc = torch.conj(self.forward_mat) 
                
            # torch.matmul does not support complexHalf
            # 1D data: [B, C, kX], 2D data: [B, C, m]
            if data.dtype == torch.complex32:
                data_inv = einsum_complexhalf('bck,bkp->bcp', data, Vc)
            else:
                data_inv = torch.matmul(data, Vc) 
        
        return data_inv

