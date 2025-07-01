import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP module supporting:
    - Conv2D/Conv3D with 1x1 kernels (`mode='conv'`)
    - Conv1D over channel dimension (`mode='channel'`)
    - Fully-connected layers (`mode='linear'`)

    Args:
        mode (str): One of ['conv', 'channel', 'linear']
        n_layers (int): Number of layers
        in_channels (int): Number of input channels/features
        out_channels (int): Output channels/features. Defaults to in_channels.
        hidden_channels (int): Hidden channels/features. Defaults to in_channels.
        n_dims (int): spatial dimension - 2 or 3 if mode='conv'. Ignored otherwise.
        non_linearity (callable or nn.Module): Activation function. Default is GELU.
        dropout (float): Dropout probability.

    Input: [batch, in_channel, nX, nY, (nZ)]
    Output: [batch, out_channel, nX, nY, (nZ)]
    """

    def __init__(self,
                 mode='channel',
                 n_layers=2,
                 in_channels=4,
                 out_channels=None,
                 hidden_channels=None,
                 n_dims=2,
                 non_linearity=nn.GELU(),
                 dropout=0.0,
                 **kwargs):
        super().__init__()

        assert mode in ['conv', 'channel', 'linear'], f"Unsupported mode: {mode}"
        assert n_layers >= 1, "n_layers must be >= 1"
        assert n_dims in [2, 3] or mode != 'conv', "n_dim must be 2 or 3 for 'conv' mode"

        self.mode = mode
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.dropout = dropout
        self.non_linearity = non_linearity

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Determine layer input/output sizes
        channels = [self.in_channels] + [self.hidden_channels] * (self.n_layers - 1) + [self.out_channels]

        for i in range(n_layers):
            in_ch, out_ch = channels[i], channels[i + 1]

            if mode == 'conv':
                Conv = nn.Conv2d if n_dims == 2 else nn.Conv3d
                layer = Conv(in_ch, out_ch, kernel_size=1)

            elif mode == 'channel':
                layer = nn.Conv1d(in_ch, out_ch, kernel_size=1)

            elif mode == 'linear':
                layer = nn.Linear(in_ch, out_ch)

            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

    def forward(self, x):
        reshaped = False
        original_shape = x.shape

        if self.mode == 'channel':
            if x.ndim > 3:
                x = x.reshape(x.shape[0], x.shape[1], -1)
                reshaped = True

        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            x = self.dropouts[i](x)

        if self.mode == 'channel' and reshaped:
            x = x.reshape((original_shape[0], self.out_channels, *original_shape[2:]))

        return x
