import os
if os.getenv("ENABLE_FLOP_WRAPPERS", "0") == "1":
    from operator_learning.utils import flop_wrappers
import torch 
import torch.nn as nn
import pandas as pd
import torch.nn.functional
from operator_learning.utils.memory_utils import CudaMemoryDebugger, format_mem
from operator_learning.utils.misc import print_rank0
from operator_learning.layers import SpectralConv, SkipConnection, GridLinear, MLP, DSELayer, NUFFTLayer
from operator_learning.data.transforms.vandermonde import VandermondeTransform
from operator_learning.data.transforms.non_uniform_fft import NUFFTTransform

class FNOLayer(nn.Module):

    def __init__(self, dv, kX, kY, kZ=None,  
                 non_linearity='gelu',
                 bias=False,
                 n_dims=2,
                 use_skip_connection=False, 
                 use_postfnochannel_mlp=False,
                 skip_type='linear',
                 use_complex_amp=False
                 ):
        super().__init__()

        self.conv = SpectralConv(dv=dv,kX=kX, kY=kY, kZ=kZ, bias=bias, dim=n_dims, use_complex_amp=use_complex_amp)
        self.use_skip_connection = use_skip_connection
        self.use_postfnochannel_mlp= use_postfnochannel_mlp
       
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)

        if use_skip_connection:
            self.skip = SkipConnection(in_channel=dv,
                                        out_channel=dv,
                                        n_dims=n_dims,
                                        skip_type=skip_type,
                                        bias=bias)
        
        if self.use_postfnochannel_mlp:
            self.channel_mlp = MLP(mode='channel',
                                   n_layers=2,
                                   n_dims=n_dims,
                                   in_channels=dv,
                                   out_channels=dv,
                                   hidden_channels=2*dv
                                )

        # self.W = GridLinear(inSize=dv,
        #                         outSize=dv,
        #                         hiddenSize=None,
        #                         bias=bias,
        #                         n_layers=1,
        #                         n_dims=n_dims,
        #                         non_linearity=self.sigma
        #                         )
        self.W = MLP( mode='channel',
                    n_dims=1,
                    n_layers=1,
                    in_channels=dv,
                    out_channels=dv,
                    hidden_channels=None,
                    )


    def forward(self, x):
        """RBC2D/3D:
          x[batchsize, dv, nX, nY, (nZ)] -> [batchsize, dv, nX, nY, (nZ)] """

        v = self.conv(x)                # Convolution
        if self.use_postfnochannel_mlp: # MLP
            v1 = self.channel_mlp(v)
            v = v + v1
        
        w = self.W(x)                   # Linear operator

        v = v + w
        if self.use_skip_connection:     # skip
            s = self.skip(x)
            v = v + s

        o = self.sigma(v)
        return o


class FNO(nn.Module):

    def __init__(self,
                 da, dv, du,
                 kX=4, kY=None, kZ=None, 
                 n_layers=2,
                 n_dims=2,
                 non_linearity='gelu',
                 bias=True, 
                 scaling_layers=4,
                 use_postfnochannel_mlp=False,
                 channel_mlp_expansion=4,
                 use_skip_connection=False, 
                 skip_type='linear',
                 use_dse=False,
                 use_toeplitz=False,
                 use_kb=False,
                 dataset=None,
                 dataClass='pic',
                 use_complex_amp=False,
                 matrix_free=False,
                 device='cpu',
                 **kwargs
                 ):
        
        super().__init__()
     
        # self.use_postfnochannel_mlp = use_postfnochannel_mlp
        self.n_dims = n_dims
        self.device = device
        self.dv = dv
        self.kX = kX
        self.kY = kY
        self.kZ = kZ

        # DSE not implemented for 3D
        self.use_dse = use_dse
        # Toeplitz cannot be implemented for PIC
        self.use_toeplitz = use_toeplitz
        # KB implemented only for PIC1D 
        self.use_kb = use_kb 
        assert sum([self.use_dse, self.use_toeplitz, self.use_kb]) <= 1, \
            "Exactly one of use_dse, use_toeplitz, or use_kb must be True."

        self.dataClass = dataClass
        self.dataset = dataset if dataClass == 'rbc' else None
        self.data_type = torch.float16 if use_complex_amp and self.training else torch.float32

        if use_dse:
            self.layers = nn.ModuleList(
                [DSELayer(dv=dv,
                          kX=kX, kY=kY, dataClass=dataClass,
                          non_linearity=non_linearity,
                          bias=bias,
                          dim=n_dims,
                          use_complex_amp=use_complex_amp,
                         )
                 for _ in range(n_layers)])
        # elif use_toeplitz:
        #     self.layers = nn.ModuleList(
        #         [NUFFTLayer(dv=dv, 
        #                   kX=kX, dataClass=dataClass,
        #                   non_linearity=non_linearity,
        #                   bias=bias,
        #                   dim=n_dims,
        #                   use_complex_amp=use_complex_amp)
        #          for _ in range(n_layers)])
        elif use_kb:
            self.layers = nn.ModuleList(
                [NUFFTLayer(dv=dv,
                          kX=kX, dataClass=dataClass,
                          non_linearity=non_linearity,
                          bias=bias,
                          dim=n_dims,
                          use_complex_amp=use_complex_amp)
                 for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList(
                [FNOLayer(dv=dv, kX=kX, kY=kY, kZ=kZ, 
                          non_linearity=non_linearity, 
                          bias=bias,
                          n_dims=n_dims,
                          use_skip_connection=use_skip_connection,
                          use_postfnochannel_mlp=use_postfnochannel_mlp,
                          skip_type=skip_type,
                          use_complex_amp=use_complex_amp)
                 for _ in range(n_layers)])
   
        self.P = MLP( mode='linear',
                        n_dims=n_dims,
                        n_layers=1,
                        in_channels=da,
                        out_channels=dv,
                        hidden_channels=round(dv*channel_mlp_expansion),
                    )
        self.Q = MLP( mode='linear',
                        n_dims=n_dims,
                        n_layers=2,
                        in_channels=dv,
                        out_channels=du,
                        hidden_channels=round(dv*channel_mlp_expansion),
                    )
       
        self.memory = CudaMemoryDebugger(print_mem=True)
 

    def forward(self, x):
        """
        RBC2D/3D:
            x[batchsize, da, nX, nY, (nZ)] -> [batchsize, du, nX, nY, (nZ)] 
        PIC1D/2D:
            x[batchsize, da, nParticle] -> [batchsize, du, nParticle]
        """

        if self.use_dse:
            if self.n_dims == 1:
                transform_coeff = VandermondeTransform(x_positions=x[:,0,:], 
                                                       kX=self.kX, 
                                                       dim=self.n_dims,
                                                       device=self.device,
                                                       dtype=self.data_type)
            elif self.n_dims == 2:
                transform_coeff = VandermondeTransform(x_positions=x[:,0,:], 
                                                       y_positions=x[:,1,:],
                                                       kX=self.kX, 
                                                       kY=self.kY,
                                                       dim=self.n_dims,
                                                       device=self.device,
                                                       dtype=self.data_type)
            else:
                raise ValueError("Vandermonde Transform not implemented for 3D")

        # if self.use_toeplitz:
        #     transform_coeff = NUFFTTransform(device=self.device,
        #                                      dataClass='pic',
        #                                      transform='toeplitz', 
        #                                      dim=self.n_dims, 
        #                                      dtype=self.data_type)
        
        if self.use_kb:
            transform_coeff = NUFFTTransform(device=self.device, 
                                             dataClass='pic',
                                             transform='kb', 
                                             dim=self.n_dims, 
                                             dtype=self.data_type)


        x = x.permute(0,2,1)
        x = self.P(x)
        x = x.permute(0,2,1)

        for index,layer in enumerate(self.layers):
            x = layer(x, transform_coeff)
          
        x = x.permute(0,2,1)
        x = self.Q(x)
        x = x.permute(0,2,1)
 

        return x

    def print_size(self):
        properties = []

        for param in self.parameters():
            properties.append([list(param.size()+(2,) if param.is_complex() else param.size()), param.numel(), (param.data.element_size() * param.numel())/1000])

        elementFrame = pd.DataFrame(properties, columns = ['ParamSize', 'NParams', 'Memory(KB)'])
        total_param = elementFrame["NParams"].sum()
        total_mem = elementFrame["Memory(KB)"].sum()
        totals = pd.DataFrame(data=[[0, total_param, total_mem]], columns=['ParamSize', 'NParams', 'Memory(KB)'])
        elementFrame = pd.concat([elementFrame,totals], ignore_index=True, sort=False)
        print_rank0(f'Total number of model parameters: {total_param} with (~{format_mem(total_mem*1000)})')
        return elementFrame

if __name__ == "__main__":
    # enable TF32 on A100 
    from operator_learning.utils.misc import enable_tf32_only_on_a100
    enable_tf32_only_on_a100()
    # Quick script testing
    model1D = FNO(da=2, dv=4, du=1, n_layers=4, kX=12, n_dims=1, use_dse=True, use_kb=False, use_toeplitz=False)
    model2D = FNO(da=3, dv=6, du=2, n_layers=4, kX=12, kY=12, n_dims=2, use_dse=True, use_kb=False, use_toeplitz=False)
    model3D = FNO(da=5, dv=10, du=5, n_layers=4, kX=12, kY=12, kZ=12, n_dims=3)
    uIn_1d = torch.rand(5, 2, 100)
    uIn_2d = torch.rand(5, 3, 100)
    uIn_3d = torch.rand(5, 5, 64, 64, 32)
    print_rank0(f"FNO1D Model Output:{model1D(uIn_1d).shape}, FNOModel: {model1D}")
    print_rank0(f"FNO2D Model Output:{model2D(uIn_2d).shape}, FNOModel: {model2D}")
    print_rank0(f"FNO3D Model Output:{model3D(uIn_3d).shape}, FNOModel: {model3D}")
