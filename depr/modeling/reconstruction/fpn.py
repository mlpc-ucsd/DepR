from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..utils.conv import GroupConv


class FPN_down_g(nn.Module):
    def __init__(self, in_channel_list, out_channel_list):
        super(FPN_down_g, self).__init__()
        self.inner_layer = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        for i, in_channel in enumerate(in_channel_list[:-1]):
            self.inner_layer.append(GroupConv(in_channel, out_channel_list[i], 1))
            self.out_layer.append(
                GroupConv(
                    out_channel_list[i], out_channel_list[i], kernel_size=3, padding=1
                )
            )

    def forward(self, x):
        features_down = []
        prev_feature = x[0]
        for i in range(len(x) - 1):
            current_feature = x[i + 1]
            prev_feature = self.inner_layer[i](prev_feature)
            size = (prev_feature.shape[2] // 2, prev_feature.shape[3] // 2)
            prev_feature = F.interpolate(prev_feature, size=size)
            prev_n_current = prev_feature + current_feature
            prev_feature = self.out_layer[i](prev_n_current)
            features_down.append(prev_feature)
        return features_down


class FPN_up_g(nn.Module):
    def __init__(self, in_channel_list, out_channel_list):
        super(FPN_up_g, self).__init__()
        self.inner_layer = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        self.depth = len(out_channel_list)
        for i, in_channel in enumerate(in_channel_list[:-1]):
            self.inner_layer.append(GroupConv(in_channel, out_channel_list[i], 1))
            self.out_layer.append(
                GroupConv(
                    out_channel_list[i], out_channel_list[i], kernel_size=3, padding=1
                )
            )

    def forward(self, x):
        features_up = []
        prev_feature = x[0]
        for i in range(self.depth):
            prev_feature = self.inner_layer[i](prev_feature)
            size = (prev_feature.shape[2] * 2, prev_feature.shape[3] * 2)
            prev_feature = F.interpolate(prev_feature, size=size)
            current_feature = x[i + 1]
            prev_n_current = prev_feature + current_feature
            prev_feature = self.out_layer[i](prev_n_current)
            features_up.append(prev_feature)

        return features_up[::-1]


class ExtraFPNBlock(nn.Module):
    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.layer_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(
        self,
        p: List[Tensor],
        c: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names
