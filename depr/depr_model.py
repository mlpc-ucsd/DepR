from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import pickle
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import ImageList
from depr.utils.inference import adaptive_load_state_dict
from .modeling.diffusers import Conv3DAwareUNet, GuidedDDIMScheduler
from .modeling.utils.conv import FrozenBatchNorm2d
from .modeling.reconstruction import (
    TriplaneVAE,
    SdfModel,
    CubeForwardProjection,
    ImagePlaneProjection,
    HighResCubeForwardProjection,
)
from .modeling.reconstruction.resnet import BasicBlock3D


@META_ARCH_REGISTRY.register()
class DepR(nn.Module):
    @configurable
    def __init__(
        self,
        backbone: nn.Module,
        reprojection: nn.Module,
        use_volume_projection: bool,
        triplane_vae: TriplaneVAE,
        scheduler,
        triplane_mlp: SdfModel,
        triplane_stat: dict[str, torch.Tensor],
        weight_dict: dict[str, float],
        diffusion_unet: Conv3DAwareUNet,
        embed_mean: float,
        embed_std: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        feature_dim: int,
        size_divisibility: int,
        cfg_prob: float,
        no_fpn: bool,
        enable_attention: bool,
        use_dino: bool,
        extrude_depth: float,
    ):
        super().__init__()

        self.backbone = backbone
        self.reprojection = reprojection
        self.triplane_vae = triplane_vae
        self.sdf_model = triplane_mlp
        self.scheduler = scheduler
        self.weight_dict = weight_dict
        self.backbone.eval()
        self.triplane_vae.eval()
        self.sdf_model.eval()
        self.cfg_prob = cfg_prob
        self.use_dino = use_dino
        self.feature_dim = feature_dim
        self.extrude_depth = extrude_depth if extrude_depth > 0.0 else None
        self.subsample_num = 7
        for param in self.parameters():
            param.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)

        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        # Trainable Parameters
        # self.diffusion_pipeline = diffusion_pipeline
        if no_fpn:
            self.fpn = None
        else:
            assert self.feature_dim == 256, "FPN is only supported with ResNet backbone"
            self.fpn = torchvision.ops.FeaturePyramidNetwork(
                [256, 512, 1024, 2048], 256
            )
            assert not self.use_dino
        self.unet = diffusion_unet
        # self.feature_project = torch.nn.Conv2d(in_channels=256, out_channels=2,kernel_size=3,stride=1,padding=1)
        self.feature_project = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.feature_dim,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.Conv2d(
                in_channels=128, out_channels=9, kernel_size=3, stride=1, padding=1
            ),
        )
        # self.feature_project = lambda x: x
        if use_volume_projection:
            self.volume_conv = torch.nn.Sequential(
                BasicBlock3D(self.feature_dim, self.feature_dim),
                BasicBlock3D(self.feature_dim, self.feature_dim),
            )
        else:
            self.volume_conv = None

        self.enable_attention = enable_attention
        if self.enable_attention:
            self.q_proj = nn.Linear(self.feature_dim, 1)
            self.cond_attn = nn.MultiheadAttention(
                embed_dim=1024,
                num_heads=8,
                dropout=0.1,
                kdim=self.feature_dim,
                vdim=self.feature_dim,
                batch_first=True,
            )
            # assert self.fpn is not None, "FPN is required for attention"
            # k,v dim is dependent on fpn

        self.surface_loss_weight = 1.0

        # For geometry loss
        self.eps = 1e-6
        self.register_buffer(
            "offset",
            torch.tensor(
                [
                    [self.eps, 0.0, 0.0],
                    [-self.eps, 0.0, 0.0],
                    [0.0, self.eps, 0.0],
                    [0.0, -self.eps, 0.0],
                    [0.0, 0.0, self.eps],
                    [0.0, 0.0, -self.eps],
                ],
            ).view(1, 1, 6, 3),
            False,
        )
        self.register_buffer("stat_range", triplane_stat["range"], False)
        self.register_buffer("stat_middle", triplane_stat["middle"], False)
        self.register_buffer("stat_max", triplane_stat["max"], False)
        self.register_buffer("stat_min", triplane_stat["min"], False)
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.skip_keys = [
            "backbone",
            "sdf_model",
            "triplane_vae",
        ]
        self.embed_std = embed_std
        self.embed_mean = embed_mean

    def load_state_dict(self, state_dict, strict=True, assign=False):
        keys_to_skip = [
            key for key in state_dict.keys() if key.split(".")[0] in self.skip_keys
        ]
        for key in keys_to_skip:
            state_dict.pop(key)
        return super().load_state_dict(state_dict, strict, assign)

    def train(self, mode=True):  # Keep frozen model in eval mode
        super().train(mode)
        self.backbone.eval()
        self.triplane_vae.eval()
        self.sdf_model.eval()
        return self

    @staticmethod
    def reset_parameters(model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif m is model:  # if m is model itself, continue
                continue
            else:
                DepR.reset_parameters(m)

    @classmethod
    def from_config(cls, cfg):
        ret = {
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "extrude_depth": cfg.MODEL.TRIPLANE_VAE.EXTRUDE_DEPTH,
            "embed_mean": cfg.MODEL.TRIPLANE_DIFFUSION.EMBED_MEAN,
            "embed_std": cfg.MODEL.TRIPLANE_DIFFUSION.EMBED_STD,
            "feature_dim": cfg.MODEL.MODEL_2D.FEATURE_DIM,
            "size_divisibility": cfg.MODEL.MODEL_2D.SIZE_DIVISIBILITY,
            "cfg_prob": cfg.MODEL.TRIPLANE_DIFFUSION.CFG_PROB,
            "backbone": build_backbone(cfg),
            "triplane_vae": TriplaneVAE(cfg),
            "diffusion_unet": Conv3DAwareUNet.from_config(
                cfg.MODEL.TRIPLANE_DIFFUSION.PRETRAINED_PATH
            ),
            "scheduler": GuidedDDIMScheduler(
                prediction_type="sample", clip_sample=False
            ),
            "no_fpn": cfg.MODEL.MODEL_2D.NO_FPN,
            "enable_attention": cfg.MODEL.TRIPLANE_DIFFUSION.ENABLE_ATTN,
            "use_dino": cfg.MODEL.BACKBONE.NAME == "build_dinov2_backbone",
        }

        if cfg.MODEL.PROJECTION == "ImagePlaneProjection":
            ret["reprojection"] = ImagePlaneProjection(cfg)
            ret["use_volume_projection"] = False
        elif cfg.MODEL.PROJECTION == "CubeForwardProjection":
            ret["reprojection"] = CubeForwardProjection(cfg)
            ret["use_volume_projection"] = True
        elif cfg.MODEL.PROJECTION == "HighResCubeForwardProjection":
            ret["reprojection"] = HighResCubeForwardProjection(cfg)
            ret["use_volume_projection"] = True
        else:
            raise ValueError(f"Unknown projection type {cfg.MODEL.PROJECTION}")

        # Load weights
        eik_weight = cfg.MODEL.TRIPLANE_DIFFUSION.EIK_WEIGHT
        nor_weight = cfg.MODEL.TRIPLANE_DIFFUSION.NOR_WEIGHT
        sur_weight = cfg.MODEL.TRIPLANE_DIFFUSION.SUR_WEIGHT
        sdf_weight = cfg.MODEL.TRIPLANE_DIFFUSION.SDF_WEIGHT
        mse_weight = cfg.MODEL.TRIPLANE_DIFFUSION.MSE_WEIGHT

        ret["weight_dict"] = {
            "loss_eik": eik_weight,
            "loss_nor": nor_weight,
            "loss_sur": sur_weight,
            "loss_sdf": sdf_weight,
            "loss_mse": mse_weight,
        }

        # Build models
        ret["triplane_mlp"] = SdfModel(
            {
                "skip_in": [],
                "n_layers": 3,
                "width": 128,
                "channels": 32,
                "ckpt_path": cfg.MODEL.TRIPLANE_VAE.MLP_PATH,
            }
        )
        assert (
            ret["scheduler"].config.prediction_type == "sample"
        ), "Only support sample prediction"

        # Load pretrained weights
        if cfg.MODEL.TRIPLANE_VAE.PRETRAINED_PATH:
            vae_state_dict = adaptive_load_state_dict(
                cfg.MODEL.TRIPLANE_VAE.PRETRAINED_PATH
            )
            ret["triplane_vae"].load_state_dict(vae_state_dict, strict=False)

        # Load stats
        with open(cfg.MODEL.TRIPLANE_VAE.STAT_PATH, "rb") as f:
            stat = pickle.load(f)
        ret["triplane_stat"] = stat

        return ret

    def normalize_embeds(self, embeds):
        return (embeds - self.embed_mean) / self.embed_std

    def denormalize_embeds(self, embeds):
        return embeds * self.embed_std + self.embed_mean

    def sample_mnfd(self, mnfd_points: torch.Tensor, mnfd_normals, num_mnfd: int):
        perm = torch.randperm(mnfd_points.size(1))
        idx = perm[:num_mnfd]
        return mnfd_points[:, idx], mnfd_normals[:, idx]

    def compute_grad(self, sdfs: torch.Tensor):
        grad = torch.cat(
            [
                0.5 * (sdfs[:, :, 0] - sdfs[:, :, 1]).unsqueeze(-1) / self.eps,
                0.5 * (sdfs[:, :, 2] - sdfs[:, :, 3]).unsqueeze(-1) / self.eps,
                0.5 * (sdfs[:, :, 4] - sdfs[:, :, 5]).unsqueeze(-1) / self.eps,
            ],
            dim=-1,
        )
        return grad

    def pad_subsamples(self, tensors: list[torch.Tensor]):
        # Using zero padding to make all tensors have self.subsample_num samples
        # tensors: list of tensors with shape (N, ...)
        # return: list of tensor with shape (self.subsample_num, ...)
        padded_tensors = []
        for tensor in tensors:
            if tensor.shape[0] == self.subsample_num:
                padded_tensors.append(tensor)
            else:
                padded_tensor = torch.zeros(
                    self.subsample_num,
                    *tensor.shape[1:],
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                padded_tensor[: tensor.shape[0]] = tensor
                padded_tensors.append(padded_tensor)
        return padded_tensors

    def denormalize_triplanes(self, triplanes: torch.Tensor):
        return (
            triplanes.view(-1, 96, 128, 128) * (self.stat_range.view(1, 96, 1, 1) / 2)
            + self.stat_middle.view(1, 96, 1, 1)
        ).view(-1, 3, 32, 128, 128)

    def compute_geometry_loss(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
        sdfs: torch.Tensor,
        normals: torch.Tensor,
        masks: torch.Tensor,
        grad_sample_ratio: float = 1.0,
    ):
        batch_size = features.shape[0]
        pcd_num = points.shape[1]
        mnfd_points = points[masks].view(batch_size, -1, 1, 3)
        mnfd_normals = normals[masks].view(batch_size, -1, 3)
        len_mnfd = mnfd_points.shape[1]
        len_unif = pcd_num - len_mnfd
        num_mnfd = int(
            mnfd_points.shape[1] * grad_sample_ratio
        )  # Only use half of the points
        num_rndm = int(len_unif * grad_sample_ratio)

        mnfd_points, mnfd_normals = self.sample_mnfd(
            mnfd_points, mnfd_normals, num_mnfd
        )
        mnfd_points = mnfd_points + self.offset
        rndm_points = (
            torch.rand(batch_size, num_rndm, 3).uniform_(-1.0, 1.0).to(features.device)
        )
        rndm_points = rndm_points.view(batch_size, -1, 1, 3) + self.offset
        denormalized_features = self.denormalize_triplanes(features)

        points_all = torch.cat(
            [
                mnfd_points.view(batch_size, -1, 3),
                rndm_points.view(batch_size, -1, 3),
                points,
            ],
            dim=1,
        )
        gt_sdfs = sdfs[~masks].view(batch_size, -1)

        # Compute SDFs
        sdfs_all = self.sdf_model(denormalized_features, points_all)

        mnfd_sdf = sdfs_all[:, : num_mnfd * 6].view(batch_size, num_mnfd, 6)
        mnfd_grad = self.compute_grad(mnfd_sdf)

        rndm_sdf = sdfs_all[:, num_mnfd * 6 : num_mnfd * 6 + num_rndm * 6].view(
            batch_size, num_rndm, 6
        )
        rndm_grad = self.compute_grad(rndm_sdf)

        pred_sdf = sdfs_all[:, num_mnfd * 6 + num_rndm * 6 :]

        rndm_grad_norm = rndm_grad.norm(2, dim=-1)
        loss_eik = nn.functional.mse_loss(
            rndm_grad_norm, torch.ones_like(rndm_grad_norm)
        )
        loss_nor = (
            nn.functional.mse_loss(mnfd_grad, mnfd_normals)
            + (1.0 - torch.sum(mnfd_grad * mnfd_normals, dim=-1)).mean()
        )
        loss_sur = nn.functional.l1_loss(
            pred_sdf[masks], torch.zeros_like(pred_sdf[masks])
        )
        loss_sdf = nn.functional.l1_loss(pred_sdf[~masks].view(batch_size, -1), gt_sdfs)
        return {
            "loss_eik": loss_eik,
            "loss_nor": loss_nor,
            "loss_sur": loss_sur,
            "loss_sdf": loss_sdf,
        }

    def project_volume_to_triplane(self, feature_3d: torch.Tensor):
        batch_size = feature_3d.shape[0]
        feature_3d = self.volume_conv(
            feature_3d.permute(0, 4, 1, 2, 3)
        )  # 4, 256, 32, 32, 32
        feature_2d = torch.stack(
            (
                feature_3d.permute(0, 1, 3, 2, 4).mean(dim=-1),  # 1, 0
                feature_3d.permute(0, 1, 4, 2, 3).mean(dim=-1),  # 2, 0
                feature_3d.permute(0, 1, 3, 4, 2).mean(dim=-1),
            ),  # 1, 2
            # feature_3d.permute(0, 1, 3, 2, 4).max(dim=-1).values,  # 1, 0
            # feature_3d.permute(0, 1, 4, 2, 3).max(dim=-1).values,  # 2, 0
            # feature_3d.permute(0, 1, 3, 4, 2).max(dim=-1).values,),# 1, 2
            dim=1,
        )  # 4, 3, 256, 32, 32
        triplane_feature = (
            self.feature_project(
                feature_2d.view(batch_size * 3, self.feature_dim, 32, 32)
            )
            .view(batch_size, 3, -1, 32, 32)
            .permute(0, 2, 3, 4, 1)
            .reshape(batch_size, -1, 32, 96)
        )  # N, C, 32, 96
        return triplane_feature

    def expand_2d_to_triplane(self, feature_2d: torch.Tensor):
        # feature_2d: N, 256, 32, 32
        triplane_feature = self.feature_project(feature_2d)  # N, C, 32, 32
        triplane_feature = F.pad(triplane_feature, (0, 64))  # N, C, 32, 96
        return triplane_feature

    @property
    def device(self):
        return self.pixel_mean.device

    def pad_masks(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(padded_masks)
        return new_targets

    def forward(
        self,
        batched_inputs,
        return_step_results=False,
        guidance_scale: float | None = None,
        return_first_step=False,
        init_noise: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            images = [
                x["image"].to(self.device) for x in batched_inputs
            ]  # B, 3, 484, 648
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(
                images, self.size_divisibility
            )  # B, 3, 512, 672 in images.tensor
            depths = ImageList.from_tensors(
                [item["depth"].to(self.device) for item in batched_inputs],
                self.size_divisibility,
            ).tensor
            features = self.backbone(images.tensor)  # res2, res3, res4, res5

        if self.fpn is None:
            encoded_features = features
        else:
            encoded_features = self.fpn(features)

        # return features, images, encoded_features
        if self.use_dino:
            lowres_features = encoded_features[0]
            highres_features = encoded_features[0]
        else:
            highres_features = encoded_features["res2"]
            lowres_features = encoded_features["res3"]

        # panoptic segmentation
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        # Instances:
        #   gt_masks: N, 484, 648
        #   gt_classes: N of int
        #   gt_is_3d_valid: N, bool
        #   gt_depths: N, 484, 648

        padded_masks = self.pad_masks(gt_instances, images)  # padding for divisibility

        intrinsic = torch.stack([target["intrinsics"] for target in batched_inputs]).to(
            self.device
        )  # B, 3, 3
        image_size = torch.as_tensor(
            [inst.image_size[::-1] for inst in gt_instances], device=self.device
        )  # B, 2

        # print(batched_inputs[0]['file_name'])
        reprojected_features = self.reprojection(
            highres_features,
            depths,
            intrinsic,
            padded_masks,
            enable_projection=(
                [instance.gt_is_3d_valid for instance in gt_instances]
                if self.training
                else None
            ),
            extrude_depth=self.extrude_depth,
        )

        if self.training:
            losses = []
            for batch_idx, reprojected_feature in enumerate(reprojected_features):
                indices = reprojected_feature["indices"]
                if len(indices) == 0:
                    continue
                sampled_indices = torch.randperm(len(indices))[: self.subsample_num]
                gt_indices = indices[sampled_indices]
                item_valid_num = sampled_indices.shape[0]

                triplane_feature = reprojected_feature["features"][sampled_indices]
                features_2d = reprojected_feature.get("features_2d")[sampled_indices]
                if self.volume_conv is not None:
                    triplane_feature = self.project_volume_to_triplane(triplane_feature)
                else:
                    triplane_feature = self.expand_2d_to_triplane(
                        triplane_feature
                    )  # N, C, 32, 96

                if self.enable_attention:
                    q_emb = self.q_proj(features_2d.permute(0, 2, 3, 1)).squeeze(
                        -1
                    )  # N, 32, 32
                    q_emb = q_emb.view(item_valid_num, -1)  # N, 1024
                    kv_emb = lowres_features[batch_idx]  # 256, 64, 84
                    kv_emb = kv_emb.view(self.feature_dim, -1).permute(
                        1, 0
                    )  # 64 * 84, 256
                    attn_feature = self.cond_attn(q_emb, kv_emb, kv_emb)[0]  # N, 1024
                    attn_feature = attn_feature.view(
                        item_valid_num, 1, 32, 32
                    )  # N, 1, 32, 32
                    attn_feature = F.pad(
                        attn_feature, (0, 64)
                    )  # pad it to N, 1, 32, 96
                    triplane_feature = torch.cat(
                        [triplane_feature, attn_feature], dim=1
                    )  # N, C+1, 32, 96

                gt_embeds = gt_instances[batch_idx].gt_embeds[
                    gt_indices
                ]  # N, 2, 32, 96
                # triplane_feature: N, C, 32, 96
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (item_valid_num,),
                    device=gt_embeds.device,
                    dtype=torch.int64,
                )  # B * N
                noise = torch.randn(gt_embeds.shape, device=gt_embeds.device)
                # Normalization, see https://github.com/huggingface/diffusers/issues/437
                noisy_embeds = self.scheduler.add_noise(
                    self.normalize_embeds(gt_embeds), noise, timesteps
                )  # N, 2, 32, 96

                drop_condition = (self.cfg_prob > 0) and (torch.rand(1) < self.cfg_prob)

                if drop_condition:
                    triplane_feature = torch.zeros_like(triplane_feature)

                input_embeds = torch.cat([noisy_embeds, triplane_feature], dim=1)
                pred_embeds = self.unet(input_embeds, timesteps).sample

                pred_embeds = self.denormalize_embeds(pred_embeds)

                with torch.no_grad():
                    pred_triplane: torch.Tensor = self.triplane_vae.decode(pred_embeds)

                # # Check if nan exists in pred_triplane or pred_embeds
                # if torch.isnan(pred_triplane).any() or torch.isnan(pred_embeds).any():
                #     print(f"NaN detected in pred_triplane or pred_embeds, skipping this batch.")
                #     continue

                item_losses = {
                    "loss_mse": F.mse_loss(pred_embeds, gt_embeds),
                    "loss_eik": 0.0,
                    "loss_nor": 0.0,
                    "loss_sur": 0.0,
                    "loss_sdf": 0.0,
                }
                if not drop_condition:
                    item_losses.update(
                        self.compute_geometry_loss(
                            pred_triplane,
                            gt_instances[batch_idx].gt_samples[gt_indices],
                            gt_instances[batch_idx].gt_sdfs[gt_indices],
                            gt_instances[batch_idx].gt_normals[gt_indices],
                            gt_instances[batch_idx].gt_masks_3d[gt_indices],
                        )
                    )

                # Check if nan exists in item_losses
                if any(
                    (
                        torch.isnan(item_losses[k]) or torch.isinf(item_losses[k])
                        if isinstance(item_losses[k], torch.Tensor)
                        else False
                    )
                    for k in item_losses
                ):
                    print(f"NaN detected in item_losses, skipping this batch.")
                    continue
                losses.append(item_losses)

            if len(losses) == 0:
                return None
            else:
                losses = {
                    k: sum([l[k] for l in losses]) / len(losses) for k in losses[0]
                }

            for k in list(losses.keys()):
                if k in self.weight_dict:
                    losses[k] *= self.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    print(f"Loss {k} not in weight_dict, removing it.")
                    losses.pop(k)
            return losses
        # Inference
        results = []
        for batch_idx, reprojected_feature in enumerate(reprojected_features):
            gt_indices = reprojected_feature["indices"]
            triplane_feature = reprojected_feature["features"]
            features_2d = reprojected_feature.get("features_2d")
            item_valid_num = triplane_feature.shape[0]
            if self.volume_conv is not None:
                triplane_feature = self.project_volume_to_triplane(triplane_feature)
            else:
                triplane_feature = self.expand_2d_to_triplane(triplane_feature)

            if self.enable_attention:
                q_emb = self.q_proj(features_2d.permute(0, 2, 3, 1)).squeeze(
                    -1
                )  # N, 32, 32
                q_emb = q_emb.view(item_valid_num, -1)  # N, 1024
                kv_emb = lowres_features[batch_idx]  # 256, 64, 84
                kv_emb = kv_emb.view(self.feature_dim, -1).permute(1, 0)  # 64 * 84, 256
                attn_feature = self.cond_attn(q_emb, kv_emb, kv_emb)[0]  # N, 1024
                attn_feature = attn_feature.view(
                    item_valid_num, 1, 32, 32
                )  # N, 1, 32, 32
                attn_feature = F.pad(attn_feature, (0, 64))  # pad it to N, 1, 32, 96
                triplane_feature = torch.cat(
                    [triplane_feature, attn_feature], dim=1
                )  # N, C+1, 32, 96

            step_results = []
            time_steps = []
            if init_noise is None:
                init_noise = torch.randn(
                    (item_valid_num, 2, 32, 96), device=triplane_feature.device
                )
            pred_embeds = init_noise
            for t in self.scheduler.timesteps:
                cond = triplane_feature
                cond_input = torch.cat((pred_embeds, cond), dim=1)
                cond_output = self.unet(cond_input, t).sample

                if guidance_scale is not None:
                    empty_cond = torch.zeros_like(cond)
                    empty_input = torch.cat((pred_embeds, empty_cond), dim=1)
                    empty_output = self.unet(empty_input, t).sample
                    model_output = empty_output + guidance_scale * (
                        cond_output - empty_output
                    )
                else:
                    model_output = cond_output

                if return_step_results:
                    step_results.append(model_output)
                    time_steps.append(t)
                if return_first_step:
                    break
                pred_embeds = self.scheduler.step(
                    model_output, t, pred_embeds
                ).prev_sample
            pred_embeds = self.denormalize_embeds(pred_embeds)
            pred_triplane = self.triplane_vae.decode(pred_embeds)
            # Taking gt_instances[batch_idx].gt_is_3d_valid into account
            results.append(
                {
                    "init_noise": init_noise,
                    "pred_embeds": pred_embeds,
                    "triplane": pred_triplane,
                    "step_results": step_results,
                    "gt_indices": gt_indices,
                    "time_steps": time_steps,
                    "triplane_feature": triplane_feature,
                }
            )
        return results
