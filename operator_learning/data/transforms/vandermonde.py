import torch
import numpy as np

# class for 2-dimensional Fourier transforms on a nonequispaced lattice of data
# ref: https://github.com/camlab-ethz/DSE-for-NeuralOperators/blob/main/ShearLayer/fno_dse.py
class VandermondeTransform:
    def __init__(self, dataset, kX, kY, device):
        self.kX = kX
        self.kY = kY

        self.x_positions = []
        self.y_positions = []
        self.Vxt_slices = []
        self.Vyt_slices = []
        self.Vxc_slices = []
        self.Vyc_slices = []

        self.device = device
        
        # logging.info("Initializing Vandermonde Transform")

        for iPatch, (sX, sY) in enumerate(dataset.slices):
            x_start, y_start = dataset.patch_startIdx[iPatch]
            padding = dataset.padding.copy()

            # Adjust padding to ensure valid indices
            padding[0] = 0 if x_start == 0 or (x_start - padding[0]) < 0 else padding[0]
            padding[1] = 0 if (x_start + sX + padding[1]) >= dataset.nX else padding[1]
            padding[2] = 0 if y_start == 0 or (y_start - padding[2]) < 0 else padding[2]
            padding[3] = 0 if (y_start + sY + padding[3]) >= dataset.nY else padding[3]

            # Extract and normalize grid positions
            xPos = torch.tensor(dataset.grid[0][x_start - padding[0]: x_start + sX + padding[1]]).clone().detach()
            yPos = torch.tensor(dataset.grid[1][y_start - padding[2]: y_start + sY + padding[3]]).clone().detach()
            
            xPos = (xPos - xPos.min()) / (xPos.max() + 1) * 2 * np.pi
            yPos = (yPos - yPos.min()) / (yPos.max() + 1) * 2 * np.pi

            self.x_positions.append(xPos)
            self.y_positions.append(yPos)
            
            # logging.info(f"Patch {iPatch}: xPos size: {len(xPos)}, yPos size: {len(yPos)}")
            
            vxt, vyt, vxc, vyc = self.make_matrix(xPos, yPos)
            self.Vxt_slices.append(vxt)
            self.Vyt_slices.append(vyt)
            self.Vxc_slices.append(vxc)
            self.Vyc_slices.append(vyc)

  
    def find_index(self, sX, sY):
        """Finds the index of the transformation matrices corresponding to a given grid size."""
        for i, (x, y) in enumerate(zip(self.x_positions, self.y_positions)):
            if len(x) == sX and len(y) == sY:
                return i
        return None
          

    def make_matrix(self, x_positions, y_positions):
        """Generates Vandermonde matrices for forward and inverse transforms."""
        sX, sY = len(x_positions), len(y_positions)

        V_x = torch.zeros([self.kX, sX], dtype=torch.cfloat).to(self.device)
        for row in range(self.kX):
             for col in range(sX):
                V_x[row, col] = torch.exp(-1j * row * x_positions[col]) 
        V_x = V_x / np.sqrt(sX)
 
        V_y = torch.zeros([2 * self.kY, sY], dtype=torch.cfloat).to(self.device)
        for row in range(self.kY):
             for col in range(sY):
                V_y[row, col] = torch.exp(-1j * row *  y_positions[col]) 
                V_y[-(row+1), col] = torch.exp(-1j * (sY - row - 1) * y_positions[col]) 
        V_y = V_y / np.sqrt(sY)

        return V_x.T, V_y.T, V_x.conj(), V_y.conj()

    def forward(self, data):
        """Computes the forward DSE transform."""
        self.idx = self.find_index(data.shape[-1], data.shape[-2])
        assert self.idx is not None, "Could not find grid for data!"
        
        data_fwd = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxt_slices[self.idx]),
                    2, 3),
                self.Vyt_slices[self.idx]),
                2, 3)

        return data_fwd
    
    def inverse(self, data):
        """Computes the inverse Fourier transform."""
        assert hasattr(self, 'idx'), "Forward transform must be called before inverse!"
        
        data_inv = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxc_slices[self.idx]),
                    2, 3),
                self.Vyc_slices[self.idx]),
                2, 3)
        
        return data_inv
