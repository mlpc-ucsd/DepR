import torch.nn as nn
from ..utils.conv import GroupConv
from ..utils.checkpoint import checkpoint
from ..utils.misc import zero_module


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, group_layer_num=32):
    return GroupNorm32(group_layer_num, channels)


class BasicBlock3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResBlock_g(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        group_layer_num_in=32,
        group_layer_num_out=32,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels, group_layer_num_in),
            nn.SiLU(),
            GroupConv(channels, self.out_channels, 3, padding=1),
            # conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, group_layer_num_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                GroupConv(self.out_channels, self.out_channels, 3, padding=1)
                # conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            # self.skip_connection = conv_nd(
            #     dims, channels, self.out_channels, 3, padding=1
            # )
            self.skip_connection = GroupConv(channels, self.out_channels, 3, padding=1)
        else:
            # self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            self.skip_connection = GroupConv(channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, [x], self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h
