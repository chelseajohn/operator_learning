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
        PIC1D/2D: x_data[nBatch, nparticle], forward_mat[nBatch, kX, nparticle]
        """
        with torch.no_grad():
            nParticle = x_data.shape[-1]

            if self.dataClass == 'pic':
                xPos = x_data.real

            # scaling btw 0 and 2*pi
            xPos = ((xPos - xPos.min()) / xPos.max()) * 2 * np.pi

            forward_mat = torch.zeros([x_data.shape[0], self.kX, nParticle], dtype=self.dtype, device=self.device)
            for row in range(self.kX):
                forward_mat[:, row, :] = torch.exp(-1j * (row - int(self.kX/2))* xPos[:, :])

            # forward_mat = torch.exp(-1j * ((self._X.to(xPos.dtype) - int(self.kX/2))* xPos[:, None, :])) # [1, kX, 1] x [nBatch, 1, nparticle]
            # norm = 1.0 / torch.sqrt(torch.tensor(nParticle, dtype=torch.int, device=self.device))
            # forward_mat.mul_(norm)

            self.forward_mat = forward_mat.to(x_data.dtype)

        return self.forward_mat
    
    def make_2Dmatrix(self, x_data, y_data):
        """
        Generates Vandermonde 2D matrices for forward and inverse transforms.
        PIC2D: x_data[nBatch, nparticle], y_data[nBatch, nparticle], forward_mat[nBatch, m, nparticle]
        """
        with torch.no_grad():
            if self.dataClass == 'pic':
                xPos = x_data.real
                yPos = y_data.real

            # scaling btw 0 and 2*pi
            xPos = ((xPos - xPos.min()) / xPos.max()) * 2 * np.pi
            yPos = ((yPos - yPos.min()) / yPos.max()) * 2 * np.pi

            m = (self.kX*2)*(self.kY*2-1)
            X_mat = torch.matmul(self._X.to(xPos.dtype), xPos[:,None,:]).repeat(1, (self.kY*2-1), 1)
            Y_mat = torch.matmul(self._Y.to(yPos.dtype), yPos[:,None,:]).repeat(1, 1, self.kX*2).reshape(yPos.shape[0], m, yPos.shape[-1]) 

            forward_mat = ((torch.exp(-1j* (X_mat+Y_mat))/ xPos.shape[-1])) 
            self.forward_mat = forward_mat.to(x_data.dtype)

        return self.forward_mat

    def forward(self, data):
        """
        Computes the forward DSE transform.
        PIC1D/2D: data[nBatch, channel, nparticle], V[nBatch, nparticle, kX or m]
        data_fwd[nBatch, channel, kX or m]
        """

        with torch.no_grad():
            if data.device != self.device:
                data = data.to(self.device)

            if self.dim == 1:
                V = self.make_1Dmatrix(x_data=data[:, 0, :]) # [nBatch, kX, nparticle]
            else:
                V = self.make_2Dmatrix(x_data=data[:,0,:], y_data=data[:,1,:])  # [nBatch, m, nparticle]
            
            # torch.bmm does not support complexHalf
            if data.dtype == torch.complex32:
                data_fwd = einsum_complexhalf('bcp,bpk->bck', data, V.permute(0,2,1))
            else:
                # data_fwd = torch.bmm(data, V.permute(0,2,1))  
                data_fwd = torch.bmm(data, V)  
            
        return data_fwd
        
    def inverse(self, data):
        """
        Computes the inverse Fourier transform.
        PIC1D/2D: data[nBatch, channel, kX or m], Vc[nBatch, kX or m, nparticle]
        data_inv[nBatch, channel, nparticle]
        """
        
        with torch.no_grad():

            Vc =  torch.conj(self.forward_mat).permute(0,2,1)
            
            # torch.matmul does not support complexHalf
            if data.dtype == torch.complex32:
                data_inv = einsum_complexhalf('bck,bkp->bcp', data, Vc)
            else:
                data_inv = torch.matmul(data, Vc) 
        
        return data_inv

