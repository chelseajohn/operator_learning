import math
import numpy as np

import torch 
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

from operator_learning.utils.memory_utils import CudaMemoryDebugger, format_mem
from operator_learning.utils.misc import print_rank0
from operator_learning.layers import SpectralConv, SkipConnection, GridLinear, MLP, DSELayer
from operator_learning.data import VandermondeTransform

class FNOLayer(nn.Module):

    def __init__(self, dv, kX, kY, kZ=None,  
                 non_linearity='gelu',
                 bias=False,
                 n_dims=2,
                 use_skip_connection=False, 
                 skip_type='linear'
                 ):
        super().__init__()

        self.conv = SpectralConv(dv=dv,kX=kX, kY=kY, kZ=kZ, bias=bias, order=n_dims)
        if non_linearity == 'gelu':
            self.sigma = nn.functional.gelu
        else:
            self.sigma = nn.ReLU(inplace=True)

        if use_skip_connection:
            self.W = SkipConnection(in_channel=dv,
                                    out_channel=dv,
                                    n_dims=n_dims,
                                    skip_type=skip_type,
                                    bias=bias)
        else:
            self.W = GridLinear(inSize=dv,
                                outSize=dv,
                                hiddenSize=None,
                                bias=bias,
                                n_layers=1,
                                n_dims=n_dims,
                                non_linearity=self.sigma
                                )


    def forward(self, x):
        """ x[nBatch, dv, nX, nY, (nZ)] -> [nBatch, dv, nX, nY, (nZ)] """

        v = self.conv(x)    # Convolution
        w = self.W(x)       # Linear operator

        v += w
        o = self.sigma(v)
        return o


class FNO(nn.Module):

    def __init__(self,
                 da, dv, du,
                 kX=4, kY=4, kZ=None, 
                 n_layers=2,
                 n_dims=2,
                 non_linearity='gelu',
                 bias=True, 
                 scaling_layers=4,
                 use_prechannel_mlp=False,
                 use_postfnochannel_mlp=False,
                 channel_mlp_expansion=4,
                 use_skip_connection=False, 
                 skip_type='linear',
                 use_dse=False,
                 get_subdomain_output=False,
                 iXBeg=0,
                 iYBeg=0,
                 iZBeg=0,
                 iXEnd=None,
                 iYEnd=None,
                 iZEnd=None,
                 dataset=None
                 ):
        
        super().__init__()
     
        self.use_postfnochannel_mlp = use_postfnochannel_mlp
        self.n_dims = n_dims
        # DSE only for 2D
        self.use_dse = use_dse
        if self.use_dse and dataset is not None:
           transformer = VandermondeTransform(dataset, kX, kY, device='cuda:0')
        else:
           transformer = None
        
        # Use conv1d
        if use_prechannel_mlp:
            self.P = MLP( mode='channel',
                          n_dims=n_dims,
                          n_layers=scaling_layers,
                          in_channels=da,
                          out_channels=dv,
                          hidden_channels=round(dv*channel_mlp_expansion),
                        )
            self.Q = MLP( mode='channel',
                          n_dims=n_dims,
                          n_layers=scaling_layers,
                          in_channels=dv,
                          out_channels=du,
                          hidden_channels=round(dv*channel_mlp_expansion),
                        )
        else:
            self.P = GridLinear(inSize=da,
                                outSize=dv,
                                hiddenSize=dv*channel_mlp_expansion,
                                bias=bias,
                                n_dims=n_dims,
                                n_layers=scaling_layers
                                )
            self.Q = GridLinear(inSize=dv,
                                outSize=du,
                                hiddenSize=dv*channel_mlp_expansion,
                                bias=bias,
                                n_dims=n_dims,
                                n_layers=scaling_layers
                                )
           
           
        if transformer is not None:
            self.layers = nn.ModuleList(
                [DSELayer(kX, kY, dv,
                          transformer,
                          non_linearity,
                          bias)
                 for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList(
                [FNOLayer(dv=dv, kX=kX, kY=kY, kZ=kZ, 
                          non_linearity=non_linearity, 
                          bias=bias,
                          n_dims=n_dims,
                          use_skip_connection=use_skip_connection,
                          skip_type=skip_type)
                 for _ in range(n_layers)])


        if self.use_postfnochannel_mlp:
            self.channel_mlp = nn.ModuleList(
                                [MLP(mode='channel',
                                    n_dims=n_dims,
                                    n_layers=scaling_layers,
                                    in_channels=dv,
                                    out_channels=dv,
                                    hidden_channels=round(dv/channel_mlp_expansion)
                                    ) 
                                for _ in range(n_layers)])
            
            self.channel_mlp_skips = nn.ModuleList(
                                        [SkipConnection(in_channel=dv,
                                                        out_channel=dv,
                                                        n_dims=n_dims,
                                                        skip_type=skip_type,
                                                        bias=bias)
                                        for _ in range(n_layers)])

        # self.memory = CudaMemoryDebugger(print_mem=True)
        self.get_subdomain_output = get_subdomain_output
        if self.get_subdomain_output:
            self.iXBeg = iXBeg
            self.iXEnd = iXEnd
            self.iYBeg = iYBeg
            self.iYEnd = iYEnd
            if self.n_dims == 3:
                self.iZBeg = iZBeg
                self.iZEnd = iZEnd

    def forward(self, x):
        """ x[nBatch, nX, nY, nZ,  da] -> [nBatch, du, nX, nY, nZ] 
            if use_subdomain_output:
                x[nBatch, nX, nY, nZ,  da] -> [nBatch, du, iXEnd-iXBeg, iYEnd-iYBeg, iZEnd-iZBeg]
        """

       
        x = self.P(x)
        # DSE only for 2D
        if self.use_dse: 
            x = x.permute(0,1,3,2).to(torch.cfloat)
            
        for index,layer in enumerate(self.layers):
            if self.use_postfnochannel_mlp:
                x_skip_channel_mlp = self.channel_mlp_skips[index](x)

            x = layer(x)

            if self.use_postfnochannel_mlp:
                 x = self.channel_mlp[index](x) + x_skip_channel_mlp
                 if index < len(self.layers) - 1:
                    x = nn.functional.gelu(x)

        if self.use_dse: 
            x = x.permute(0,1,3,2).real
        
        # to get only a subdomain output inference
        if self.get_subdomain_output:
            print_rank0(f'Filtering to x-subdomain {self.iXBeg,self.iXEnd} & y-subdomain {self.iYBeg,self.iYEnd}')
            x = x[:, :, self.iXBeg:self.iXEnd, self.iYBeg:self.iYEnd]
            if self.n_dims == 3:
                print_rank0(f' & z-subdomain {self.iZBeg, self.iZEnd} ')
                x = x [:, :, :, :, self.iZBeg: self.iZEnd]

        x = self.Q(x)
        # print_rank0(f'Shape of x: {x.shape}')

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
    # Quick script testing
    model = FNO(da=4, dv=4, du=4, n_layers=4, kX=12, kY=12)
    uIn = torch.rand(5, 4, 256, 64)
    print_rank0(model(uIn).shape)
