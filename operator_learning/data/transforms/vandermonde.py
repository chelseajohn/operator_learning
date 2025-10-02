import torch
import numpy as np

# class for 1,2-dimensional Fourier transforms on a nonequispaced lattice of data
# ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py
class VandermondeTransform:
    def __init__(self, kX, kY=None, dataset=None, dim=1):
        self.kX = kX
        self.kY = kY
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        assert dataset is not None, "dataset is required for position normalization"
  
        try:
            xPos = torch.tensor(dataset.grid[0])
            self.nCol = xPos.shape[0]
        except AttributeError:
            if dim == 1:
                xPos = torch.tensor(dataset.inputs[:, 0, :].flatten(), dtype=torch.float32)
                self.nCol = dataset.inputs.shape[-1]
            else:
                xPos = torch.tensor(dataset.inputs[:, 0, :, 0].flatten(), dtype=torch.float32)
                self.nCol = dataset.inputs.shape[-2]
      
        # scaling btw 0 and 2*pi
        self.xPos = ((xPos - xPos.min()) / xPos.max()) * 2 * np.pi
        self._X =  torch.arange(self.kX, dtype=torch.float)[None, :, None] 

        if dim == 2 and kY is not None:
            try:
                yPos = torch.tensor(dataset.grid[1], dtype=torch.float32)
                self.nRow = yPos.shape[0]
            except AttributeError:
                yPos = torch.tensor(dataset.inputs[:, 0, 0, :].flatten(), dtype=torch.float32)
                self.nRow = dataset.inputs.shape[-1]
    
            # scaling btw 0 and 2*pi
            self.yPos = ((yPos - yPos.min()) / yPos.max()) * 2 * np.pi
            self.nRow = dataset.inputs.shape[-1]
            self._Y = torch.arange(self.kY, dtype=torch.float)[None, :, None]

    def make_1Dmatrix(self, x_data):
        """Generates Vandermonde 1D matrices for forward and inverse transforms."""
        # x_data: [B, nCol=nX], V: [B, kX, nX]
        V = torch.exp(-1j * self._X * x_data[:, None, :])  # [1, kX, 1] x [B, kX, nX]
        self.V = V /np.sqrt(self.nCol)  # normalization

        return  self.V.clone().permute(0,2,1)# [B, nX, kX]
    
    def make_2Dmatrix(self, x_data, y_data):
        """Generates Vandermonde 2D matrices for forward and inverse transforms."""
        # x_data: [B, nX], y_data: [B, nY], V_x: [B, kY, nY], V_y: [B, 2*kX, nX]
        V_x = torch.exp(-1j * self._Y * y_data[:, None, :])  # [1, kY, 1] x [B, kY, nY]
        self.V_x = V_x / np.sqrt(self.nRow)
 
        V_ytop = torch.exp(-1j * self._X * x_data[:, None, :]) # [1, kX, 1] x [B, kX, nX]
        V_ybot = torch.exp(-1j * (self.nCol - self._X - 1) * x_data[:,None,:]) # [1, kX, 1] x [B, kX, nX]
        V_y = torch.cat([V_ytop, V_ybot], dim=1)  # [B, 2*kX, nX]
        self.V_y = V_y / np.sqrt(self.nCol)

       
        return self.V_x.clone().permute(0,2,1), self.V_y.clone().permute(0,2,1) # [B, nX(nY), kX(kY)]
    
    def forward(self, data):

        """Computes the forward DSE transform."""

        if self.dim == 1:
           # data: [B, C, nX], Vt: [B, nX, kX]
           Vt  = self.make_1Dmatrix(x_data=data[:, 0, :])
           data_fwd = torch.bmm(data, Vt.to(data.device))
        else:
            # data: [B, C, nX, nY], Vxt: [B, nY, kY], Vyt: [nX, 2*kX]
            Vxt, Vyt = self.make_2Dmatrix(x_data=data[:,0,:,0], y_data=data[:,0,0,:])
            x = torch.matmul(data, Vxt.to(data.device))  # [B, C, nX, kY]
            Vyt = Vyt.transpose(-2, -1)  # [2*kX, nX]
            data_fwd = torch.matmul(Vyt.to(data.device), x)  # [B, C, 2*kX, kY]


        return data_fwd
    
    def inverse(self, data):
        """Computes the inverse Fourier transform."""
        if self.dim == 1:
            # data: [B, C, kX], Vc: [B, kX, nX]
            Vc =  torch.conj(self.V.clone())
            data_inv = torch.bmm(data, Vc.to(data.device))
        else:
            # data: [B, C, 2*kX, kY], Vxc: [kY, nY], Vyc: [2*kX, nX]
            Vxc = torch.conj(self.V_x.clone())
            Vyc = torch.conj(self.V_y.clone())
            x = torch.matmul(data, Vxc.to(data.device)) # [B, C, 2*kX, nY]
            x = x.transpose(-2,-1)  # [B, C, nY, 2*kX]
            data_inv =  torch.matmul(x, Vyc.to(data.device)) # [B,C, nY, nX]
            data_inv = data_inv.transpose(-2,-1) # [B,C, nX, NY]
   
        return data_inv

