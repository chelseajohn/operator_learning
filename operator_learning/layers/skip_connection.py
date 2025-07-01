import torch
import torch.nn as nn

class SkipConnection(nn.Module):
    """
    SkipConnection supports 'identity', 
    'linear', and 'soft-gating' skip types.
    
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        n_dim (int): Spatial dimensions (2 for Conv2D, 3 for Conv3D, etc).
        skip_type (str): One of {'identity', 'linear', 'soft-gating'}.
        bias (bool): Include bias in linear/soft-gating transforms.
    """

    def __init__(self, 
                in_channel, 
                out_channel,
                n_dim=2,
                skip_type="soft-gating",
                bias=False):
        super().__init__()

        assert skip_type in {"identity", "linear", "soft-gating"}, f"Invalid skip_type: {skip_type}"
        self.skip_type = skip_type.lower()

        if self.skip_type == "identity":
            self.skip = nn.Identity()

        elif self.skip_type == "soft-gating":
            # Input: [bacthsize, channels, nX, nY, (nZ)]
            # Output: Input * [1, channels, 1, 1, (1)]
            if in_channel != out_channel:
                raise ValueError(
                    f"Soft-gating requires in_channel == out_channel, "
                    f"but got {in_channel} != {out_channel}"
                )
            shape = (1, in_channel) + (1,) * n_dim  # e.g., (1, C, 1, 1) for 2D
            self.weight = nn.Parameter(torch.ones(shape))
            self.bias = nn.Parameter(torch.ones(shape)) if bias else None

        elif self.skip_type == "linear":
            # For Input shape > 3
            # Flattens all dimensions after bacthsize and channel
            # Applies 1D conv then un-flattens
            self.conv = nn.Conv1d(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=1,
                                  bias=bias)

    def forward(self, x):
        if self.skip_type == "identity":
            return self.skip(x)

        elif self.skip_type == "soft-gating":
            return x * self.weight + self.bias if self.bias is not None else x * self.weight

        elif self.skip_type == "linear":
            size = x.shape
            x = x.view(size[0], size[1], -1)  # flatten spatial dims
            x = self.conv(x)
            x = x.view(size[0], self.conv.out_channels, *size[2:])  # reshape back
            return x
