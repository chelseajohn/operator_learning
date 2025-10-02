import torch
import numpy as np
from typing import List

# class for 1,2-dimensional Fourier transforms on a nonequispaced lattice of data
# ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py
class VandermondeTransform:
    def __init__(self, kX, kY=None, dataset=None, position: List[np.ndarray] = None, dim=2):
        self.kX = kX
        self.kY = kY
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        assert (dataset is None) ^ (position is None), \
            "Provide exactly one of dataset or positions (not both or none)"

        if dataset is not None:
            try:
                xPos = torch.tensor(dataset.grid[0])
            except AttributeError:
                if dim == 1:
                    xPos = torch.tensor(dataset.inputs[:, 0, :].flatten(), dtype=torch.float32)
                else:
                    xPos = torch.tensor(dataset.inputs[:, 0, :, 0].flatten(), dtype=torch.float32)
        else:
            xPos = torch.tensor(position[0], dtype=torch.float32)
    
        # scaling btw 0 2*pi
        self.xPos = (xPos - xPos.min()) / (xPos.max() + 1) * 2 * np.pi
        self.nCol = self.xPos.shape[0]

        if dim == 2 and kY is not None:
            if dataset is not None:
                try:
                    yPos = torch.tensor(dataset.grid[1], dtype=torch.float32)
                except AttributeError:
                    yPos = torch.tensor(dataset.inputs[:, 0, 0, :].flatten(), dtype=torch.float32)
            else:
                yPos = torch.tensor(position[1], dtype=torch.float32)
            
            # scaling btw 0 2*pi
            self.yPos = (yPos - yPos.min()) / (yPos.max() + 1) * 2 * np.pi
            self.nRow = self.yPos.shape[0]

        if dim == 1:
            self.Vt, self.Vc = self.make_1Dmatrix()
        else:
            self.Vxt, self.Vxc, self.Vyt, self.Vyc = self.make_2Dmatrix()

    def make_1Dmatrix(self):
        """Generates Vandermonde 1D matrices for forward and inverse transforms."""
        V = torch.zeros([self.kX, self.nCol], dtype=torch.cfloat)
        for row in range(self.kX):
            for col in range(self.nCol):
                V[row, col] = torch.exp(-1j * row * self.xPos[col])
        
        V = V /np.sqrt(self.nCol)  # normalization

        V_inv = torch.conj(V.clone())

        return V.T, V_inv
    
    def make_2Dmatrix(self):
        """Generates Vandermonde 2D matrices for forward and inverse transforms."""

        V_x = torch.zeros([self.kY, self.nRow], dtype=torch.cfloat)
        for row in range(self.kY):
             for col in range(self.nRow):
                V_x[row, col] = torch.exp(-1j * row * self.yPos[col]) 
        V_x = V_x / np.sqrt(self.nRow)
 
        V_y = torch.zeros([2 * self.kX, self.nCol], dtype=torch.cfloat)
        for row in range(self.kX):
             for col in range(self.nCol):
                V_y[row, col] = torch.exp(-1j * row *  self.xPos[col]) 
                V_y[-(row+1), col] = torch.exp(-1j * (self.nCol - row - 1) * self.xPos[col]) 
        V_y = V_y / np.sqrt(self.nCol)

        return V_x.T, torch.conj(V_x.clone()), V_y.T, torch.conj(V_y.clone())
    
    def forward(self, data):

        """Computes the forward DSE transform."""

        if self.dim == 1:
           data_fwd = torch.matmul(data, self.Vt.to(data.device))
        else:
            # data: [B, C, Nx, Ny], Vxt: [Ny, Ky], Vyt: [Nx, 2*Kx]
            x = torch.matmul(data, self.Vxt.to(data.device))  # [B,C,Nx,Ky]
            x = x.transpose(-2, -1)  # [B,C,Ky,Nx]
            data_fwd = torch.matmul(x, self.Vyt.to(data.device))  # [B,C,Ky,2*Kx]
            data_fwd = data_fwd.transpose(-2,-1) # [B,C,2*Kx,Ky]

        return data_fwd
    
    def inverse(self, data):
        """Computes the inverse Fourier transform."""
        if self.dim == 1:
            data_inv = torch.matmul(data, self.Vc.to(data.device))
        else:
            # data: [B, C, 2*Kx, Ky], Vxc: [Ky, Ny], Vyc: [2*Kx, Nx]
            x = torch.matmul(data, self.Vxc.to(data.device)) # [B,C,2*Kx,Ny]
            x = x.transpose(-2,-1)  # [B,C,Ny,2*Kx]
            data_inv =  torch.matmul(x,self.Vyc.to(data.device)) # [B,C,Ny, Nx]
            data_inv = data_inv.transpose(-2,-1) # [B,C,Nx, NY]
   
        return data_inv

