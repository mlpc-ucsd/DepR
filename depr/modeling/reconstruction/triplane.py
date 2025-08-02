import torch
import torch.nn as nn
from torch import nn, Tensor
from torch.jit.annotations import List
from einops import reduce
from .resnet import ResBlock_g
from .fpn import FPN_down_g, FPN_up_g
from .transformer import SpatialTransformer
from ..utils.conv import GroupConvTranspose, GroupConv


class TriplaneVAE(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        kl_std = cfg.MODEL.TRIPLANE_VAE.KL_STD
        plane_shape = cfg.MODEL.TRIPLANE_VAE.PLANE_SHAPE
        z_shape = cfg.MODEL.TRIPLANE_VAE.Z_SHAPE
        num_heads = cfg.MODEL.TRIPLANE_VAE.NUM_HEADS
        transform_depth = cfg.MODEL.TRIPLANE_VAE.TRANSFORM_DEPTH

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape

        self.kl_std = kl_std

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2 * self.z_shape[0]]
        # feature size:  64,  32,  16,   8,    4,    8,   16,       32
        feature_size = [64, 32, 16, 8, 4, 8, 16, 32]
        #

        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        # feature size:  16,    8,   4,   8,    16,  32,  64

        self.in_layer = nn.Sequential(
            ResBlock_g(
                32,
                dropout=0,
                out_channels=128,
                use_conv=True,
                dims=2,
                use_checkpoint=False,
                group_layer_num_in=1,
            ),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        #
        # self.spatial_modulation = nn.Linear(128*3, 128*3)

        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    GroupConv(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock_g(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                ),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))

        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(
                nn.Sequential(
                    # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    GroupConv(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    SpatialTransformer(
                        h_dim,
                        num_heads,
                        dim_head,
                        depth=transform_depth,
                        context_dim=h_dim,
                        disable_self_attn=False,
                        use_linear=True,
                        attn_type="linear",
                        use_checkpoint=False,
                        layer=feature_size[i + 1],
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                )
            )
            in_channels = h_dim

        self.encoder_fpn = FPN_down_g([512, 512, 1024, 1024], [512, 1024, 1024])
        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(
                nn.Sequential(
                    GroupConvTranspose(
                        in_channels,
                        h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                )
            )
            if i == 2:
                modules.append(
                    nn.Sequential(
                        ResBlock_g(
                            h_dim,
                            dropout=0,
                            out_channels=2 * z_shape[0],
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1,
                            group_layer_num_out=1,
                        ),
                        nn.BatchNorm2d(2 * z_shape[0]),
                        nn.SiLU(),
                    )
                )
                in_channels = z_shape[0]
            else:
                modules.append(
                    nn.Sequential(
                        SpatialTransformer(
                            h_dim,
                            num_heads,
                            dim_head,
                            depth=transform_depth,
                            context_dim=h_dim,
                            disable_self_attn=False,
                            use_linear=True,
                            attn_type="linear",
                            use_checkpoint=False,
                            layer=feature_size[i + 5],
                        ),
                        nn.BatchNorm2d(h_dim),
                        nn.SiLU(),
                    )
                )
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        # feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoder_in_layer = nn.Sequential(
            ResBlock_g(
                self.z_shape[0],
                dropout=0,
                out_channels=512,
                use_conv=True,
                dims=2,
                use_checkpoint=False,
                group_layer_num_in=1,
            ),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(
                nn.Sequential(
                    # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    GroupConv(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    SpatialTransformer(
                        h_dim,
                        num_heads,
                        dim_head,
                        depth=transform_depth,
                        context_dim=h_dim,
                        disable_self_attn=False,
                        use_linear=True,
                        attn_type="linear",
                        use_checkpoint=False,
                        layer=feature_size_decoder[i],
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                )
            )
            in_channels = h_dim

        # self.decoder_fpn = FPN_up([1024, 1024, 1024, 512], [1024, 1024, 512])
        self.decoder_fpn = FPN_up_g([1024, 1024, 1024, 512], [1024, 1024, 512])
        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(
                nn.Sequential(
                    GroupConvTranspose(
                        in_channels,
                        h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                )
            )
            if i < 4:
                modules.append(
                    nn.Sequential(
                        SpatialTransformer(
                            h_dim,
                            num_heads,
                            dim_head,
                            depth=transform_depth,
                            context_dim=h_dim,
                            disable_self_attn=False,
                            use_linear=True,
                            attn_type="linear",
                            use_checkpoint=False,
                            layer=feature_size_decoder[i + 3],
                        ),
                        nn.BatchNorm2d(h_dim),
                        nn.SiLU(),
                    )
                )
                in_channels = h_dim
            else:
                modules.append(
                    nn.Sequential(
                        ResBlock_g(
                            h_dim,
                            dropout=0,
                            out_channels=h_dim,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                        ),
                        nn.BatchNorm2d(h_dim),
                        nn.SiLU(),
                    )
                )
                in_channels = h_dim
            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(
            nn.Sequential(
                GroupConvTranspose(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
                ResBlock_g(
                    in_channels,
                    dropout=0,
                    out_channels=self.plane_shape[1],
                    use_conv=True,
                    dims=2,
                    use_checkpoint=False,
                ),
                nn.BatchNorm2d(self.plane_shape[1]),
                nn.Tanh(),
            )
        )

    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        result = enc_input
        if self.plane_dim == 5:
            result = torch.concat(torch.chunk(result, 3, dim=1), dim=-1).squeeze(1)
        elif self.plane_dim == 4:
            result = torch.concat(torch.chunk(result, 3, dim=1), dim=-1)

        feature = self.in_layer(result)
        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [0, 1, 2, 3]:
                features_down.append(feature)

        features_down = self.encoder_fpn(features_down)

        # breakpoint()

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            feature_down.append(x)
        feature_down = self.decoder_fpn(feature_down[::-1])
        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)
        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat(
                [
                    x[..., 0:plane_w].unsqueeze(1),
                    x[..., plane_w : plane_w * 2].unsqueeze(1),
                    x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),
                ],
                dim=1,
            )
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat(
                [
                    x[..., 0:plane_w],
                    x[..., plane_w : plane_w * 2],
                    x[..., plane_w * 2 : plane_w * 3],
                ],
                dim=1,
            )
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)

        return [result, data, mu, log_var, z]

    # only using VAE loss
    def loss_function(self, *args) -> dict:
        mu = args[2]
        log_var = args[3]

        if self.kl_std == "zero_mean":
            latent = self.reparameterize(mu, log_var)
            # print("latent shape: ", latent.shape) # (B, dim)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]

        else:
            std = torch.exp(torch.clamp(0.5 * log_var, max=10)) + 1e-6
            gt_dist = torch.distributions.normal.Normal(
                torch.zeros_like(mu), torch.ones_like(std) * self.kl_std
            )
            sampled_dist = torch.distributions.normal.Normal(mu, std)
            # gt_dist = normal_dist.sample(log_var.shape)
            # print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(
                sampled_dist, gt_dist
            )  # reversed KL
            kl_loss = reduce(kl, "b ... -> b (...)", "mean").mean()

        return kl_loss

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        gt_dist = torch.distributions.normal.Normal(
            torch.zeros(num_samples, *(z_rollout_shape)),
            torch.ones(num_samples, *(z_rollout_shape)) * self.kl_std,
        )

        z = gt_dist.sample().cuda()
        samples = self.decode(z)
        return samples, z

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_latent(self, x):
        """
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z
