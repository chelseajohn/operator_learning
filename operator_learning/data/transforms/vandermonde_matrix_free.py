import torch
from operator_learning.utils.misc import einsum_complexhalf

class VandermondeTransformMatrixFree:
    """
    Matrix-free 1D/2D Fourier transforms on a nonequispaced lattice.
    Memory complexity: O(batch * nPoints) instead of O(batch * modes * nPoints).
    """
    def __init__(self, x_positions, kX, y_positions=None, kY=None, dim=1, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        assert dim in (1, 2), "dim must be 1 or 2"
        self.dim = dim
        self.kX = kX
        self.batch_size = x_positions.shape[0]
        self.number_points = x_positions.shape[1]

        with torch.no_grad():
            x_positions -= torch.min(x_positions)
            self.x_positions = (x_positions * 6.28 / torch.max(x_positions)).to(device)  # [batchsize, nParticle]
        
            if dim == 1:
                # [kX] — centered at zero
                self.kX_vec = torch.arange(kX, dtype=dtype, device=device) - int(kX / 2)  
            else:
                self.kY = kY if kY is not None else kX
                y_positions -= torch.min(y_positions)
                self.y_positions = (y_positions * 6.28 / torch.max(y_positions)).to(device)  # [batchsize, nParticle]

                # X: [0, ..., kX-1, -kX, ..., -1] -> [2*kX]
                self.kX_vec = torch.cat([
                    torch.arange(kX, dtype=dtype, device=device),
                    torch.arange(start=-kX, end=0, dtype=dtype, device=device)
                ]) 

                # Y: [0, ..., kY-1, -(kY-1), ..., -1]->[2*kY]
                self.kY_vec = torch.cat([
                    torch.arange(kY, dtype=dtype, device=device),
                    torch.arange(start=-kY, end=0, dtype=dtype, device=device)
                ])  # [2*kY]

    def _make_1d_matrix(self):
        """Build and cache the matrix once, outside autograd."""
        # [1, kX, 1] * x_positions [batchsize, 1, nParticle]
        self.forward_mat = torch.exp(
            -1j * self.kX_vec[None, :, None] * self.x_positions[:, None, :]
        ).detach()  # [batchsize, kX, nParticle] — detach so it's not part of any graph

    def _make_2d_matrices(self):
        """Build and cache y-phases once. x-phases are cheap per-iteration."""

        # y-phase [batchsize, mY, nParticle]  — constant across kx loop
        self.y_phases_fwd = torch.exp(
            -1j * self.kY_vec[None, :, None] * self.y_positions[:, None, :]
            ).detach()  

        self.y_phases_inv = self.y_phases_fwd.conj().detach()  

    def _forward_1d(self, data):
        """
        data: [batchsize, dv, nParticle]  (real)
        out:  [batchsize, dv, kX]  (complex)
        Computes out[batchsize, channel, modes] = (sum_n exp(-j * kX_vec[k] * x[batchsize, nParticle]) 
                                                * data[batchsize, dv, nParticle])
        without forming the [batchsize, kX, nParticle] Vandermonde matrix.
        """
        
        if not hasattr(self, 'forward_mat'):
            self._make_1d_matrix()

        # [batchsize, dv, nParticle] x [batchsize, nParticle, kX] 
        if data.dtype == torch.complex32:
            return einsum_complexhalf('bcp,bpk->bck', data, self.forward_mat.permute(0, 2, 1))
        else:
            return torch.bmm(data, self.forward_mat.permute(0, 2, 1))

    def _inverse_1d(self, data):
        """
        data: [batchsize, dv, kX]
        out:  [batchsize, dv, nParticle]
        Uses conjugate phases.
        """
        if not hasattr(self, 'forward_mat'):
            self._make_1d_matrix()

        # [batchsize, dv, kX] x [batchsize, kX, nParticle] 
        if data.dtype == torch.complex32:
            return einsum_complexhalf('bck,bkp->bcp', data, self.forward_mat.conj()) 
        else:
            return torch.bmm(data, self.forward_mat.conj()) 


    def _forward_2d(self, data):
        """
        data: [batchsize, dv, nParticle]
        out:  [batchsize, dv, (2*kX)*(2*kY)]
        Computes exp(-j*(kx*x + ky*y)) @ data without storing the full matrix.

        Strategy: iterate over kX modes (outer loop kept small = 2*kX iterations),
        accumulating [batchsize, dv, 2*kY] slice per kx step.
        Peak extra memory: O(B * N * (2*kY))  —  one phase strip at a time.
        """
        
        if not hasattr(self, 'y_phases_fwd'):
            self._make_2d_matrices()
        
        B, dv, _ = data.shape
        mX = 2 * self.kX
        mY = 2 * self.kY 

        out = torch.zeros(B, dv, mX * mY, dtype=torch.cfloat, device=self.device)
      
        for i, kx in enumerate(self.kX_vec):
            x_phase = torch.exp(-1j * kx * self.x_positions[:, None, :]).detach()  # [batchsize, 1, nParticle]

            # combined phase strip: [batchsize, mY, nParticle]
            V = self.y_phases_fwd * x_phase  # broadcast over mY

            # [batchsize, dv, nParticle] x [batchsize, nParticle, mY] 
            if data.dtype == torch.complex32:
                data_fwd = einsum_complexhalf('bcp,bpk->bck', data, V.permute(0,2,1))
            else:
                data_fwd = torch.bmm(data, V.permute(0, 2, 1))

            out[:, :, i * mY:(i + 1) * mY] = data_fwd
            del V, data_fwd, x_phase 

        return out / self.number_points

    def _inverse_2d(self, data):
        """
        data: [batchsize, dv, (2*kX)*(2*kY)]
        out:  [batchsize, dv, nParticle]
        """
        if not hasattr(self, 'y_phases_inv'):
            self._make_2d_matrices()

        B, dv, _ = data.shape
        mY = 2 * self.kY 

        out = torch.zeros(B, dv, self.number_points, dtype=torch.cfloat, device=self.device)

        for i, kx in enumerate(self.kX_vec):
            x_phase_conj = torch.exp(1j * kx * self.x_positions[:, None, :]).detach() # [batchsize, 1, nParticle]
            Vc = self.y_phases_inv * x_phase_conj  # [batchsize, mY, nParticle]

            # [batchsize, dv, mY] x [batchsize, mY, nParticle]
            coeff_slice = data[:, :, i * mY:(i + 1) * mY] 
            if data.dtype == torch.complex32:
                data_inv = einsum_complexhalf('bck,bkp->bcp', coeff_slice, Vc)
            else:
                data_inv = torch.bmm(coeff_slice, Vc)
            out += data_inv
            del Vc, x_phase_conj, data_inv

        return out

    def forward(self, data):
        """
        data: [batchsize, dv, nParticle]
        returns: [batchsize, dv, modes]
        """
        if data.device.type != self.device:
            data = data.to(self.device)

        if self.dim == 1:
            return self._forward_1d(data)
        else:
            return self._forward_2d(data)

    def inverse(self, data):
        """
        data: [batchsize, dv, modes]
        returns: [batchsize, dv, nParticle]
        """
        if self.dim == 1:
            return self._inverse_1d(data)
        else:
            return self._inverse_2d(data)