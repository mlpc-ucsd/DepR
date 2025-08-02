import torch
import torch.nn as nn
from detectron2.layers import FrozenBatchNorm2d as FBN2d


class FrozenBatchNorm2d(FBN2d):
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # We revert the behavior to the original PyTorch implementation
        return super(FBN2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class GroupConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ) -> None:
        super(GroupConv, self).__init__()
        self.conv = nn.Conv2d(
            3 * in_channels, 3 * out_channels, kernel_size, stride, padding, groups=3
        )

    def forward(self, data):
        data = torch.concat(torch.chunk(data, 3, dim=-1), dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data, 3, dim=1), dim=-1)
        return data


class GroupConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
    ) -> None:
        super(GroupConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose2d(
            3 * in_channels,
            3 * out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups=3,
        )

    def forward(self, data):
        data = torch.concat(torch.chunk(data, 3, dim=-1), dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data, 3, dim=1), dim=-1)
        return data
