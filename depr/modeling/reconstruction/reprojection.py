import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
import copy


def crop_roi(feature, mask, size):
    mask_indices = torch.nonzero(mask, as_tuple=False)
    x_1, y_1 = mask_indices.min(dim=0)[0]
    x_2, y_2 = mask_indices.max(dim=0)[0]

    # make box be square
    x_len = x_2 - x_1
    y_len = y_2 - y_1
    diff = abs(x_len - y_len)
    if x_len > y_len:
        y_1 -= diff // 2
        y_2 += diff // 2
    else:
        x_1 -= diff // 2
        x_2 += diff // 2

    boxes = [torch.tensor([[y_1, x_1, y_2, x_2]]).float().cuda()]
    cropped_feature = torchvision.ops.roi_align(
        feature.unsqueeze(0).float(), boxes, size
    )[0]
    return cropped_feature


def crop_roi_inter(
    image: torch.Tensor, mask: torch.Tensor, size: int, mode: str
) -> torch.Tensor:
    # Find the coordinates where mask is True
    mask_indices = torch.nonzero(mask, as_tuple=False)

    # Get the min and max coordinates for both dimensions
    y_min, x_min = mask_indices.min(dim=0)[0]
    y_max, x_max = mask_indices.max(dim=0)[0]

    # Calculate the side length of the square that contains the mask
    side_length = max(y_max - y_min, x_max - x_min)

    # Determine the square crop area
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # Calculate the start and end points for the square crop
    new_y_min = max(center_y - side_length // 2, 0)
    new_x_min = max(center_x - side_length // 2, 0)
    new_y_max = new_y_min + side_length
    new_x_max = new_x_min + side_length

    # Pad the image if necessary
    pad_y = max(0, new_y_max - image.shape[1])
    pad_x = max(0, new_x_max - image.shape[2])
    padded_image = F.pad(image, (0, pad_x, 0, pad_y))

    # Crop the padded image
    cropped_image = padded_image[:, new_y_min:new_y_max, new_x_min:new_x_max]

    # Resize the image to the desired size (32x32)
    resized_image = F.interpolate(
        cropped_image.unsqueeze(0), size=(size, size), mode=mode
    )

    return resized_image.squeeze(0)


"""
A backward projection module that project 3D features to 2D triplane features
In opposite to forward projection, which projects 2D triplane features to 3D triplane features
Input: batched_inputs
Output: batched_triplanes

Steps:
1. Project the depth map to camera coordinates
2. For each object in the scene, get the 3D bounding box
3. Normalize the 3D bounding box to a cube
4. Project the camera coordinates on the triplanes

# 3. For each 3D bounding box, get its triplane cube and corresponding 3D points in grid
# 4. For each 3D point, get its 2D projection in the image
# 5. Use grid_sample to get the 2D projection of the feature for triplane features

Config:
1. triplane dimensions: 32x32
"""


class CubeForwardProjection(nn.Module):
    @configurable
    def __init__(self, triplane_dims=32):
        super().__init__()
        self.triplane_dims = triplane_dims

    @classmethod
    def from_config(cls, cfg):
        return {
            "triplane_dims": cfg.MODEL.TRIPLANE_VAE.DIMENSIONS,
        }

    def forward(
        self, encoder_features, depths, intrinsics, masks, enable_projection=None
    ):
        """
        Args:
            encoder_features: (N, C, H, W) tensor of encoder features

        Returns:
            batched_triplanes: list of dictionaries containing: triplane features, cube center, cube size
        """
        batched_triplanes = []
        for batch_id in range(len(depths)):
            encoder_feature = encoder_features[batch_id]  # C, H, W: 256, 128, 168
            C, H, W = encoder_feature.shape
            depth = depths[batch_id]  # H, W
            intrinsic = copy.deepcopy(intrinsics[batch_id])  # 3, 3
            segments = masks[batch_id]  # List of masks
            if len(segments) == 0:
                batched_triplanes.append({})
                continue

            intrinsic[2, 2] *= 4
            # Get camera coords
            intrinsic_inverse = torch.inverse(
                intrinsic
            )  # The divisibility padding is applied to down right, so the intrinsic wouldn't change
            # )  # Align depth with features
            feature_depth = depth[::4, ::4]  # Downsample
            feature_points = project_depth_to_camera_coordinates(
                feature_depth, intrinsic_inverse
            )

            result_grid = []
            valid_indices = []
            result_feature_2d = []
            for idx, mask in enumerate(segments):
                if (enable_projection is not None) and (
                    not enable_projection[batch_id][idx]
                ):
                    continue
                segment_mask = mask[::4, ::4]  # Downsample to feature map size
                segment_mask = segment_mask & (feature_depth > 0)
                if segment_mask.sum() < 10:
                    continue

                masked_features = encoder_feature.clone().detach()
                masked_features[:, ~segment_mask] = 0
                roi_features_2d = crop_roi_inter(
                    masked_features, segment_mask, self.triplane_dims, mode="bilinear"
                )  # 256, 32, 32
                roi_features = roi_features_2d.permute(1, 2, 0).view(-1, C)  # N, 256
                masked_points = feature_points.clone().detach()  # H, W, 3
                masked_points[~segment_mask] = 0
                roi_points = crop_roi_inter(
                    masked_points.permute(2, 0, 1),
                    segment_mask,
                    self.triplane_dims,
                    mode="nearest",
                )
                roi_points = roi_points.permute(1, 2, 0).view(-1, 3)  # N, 3

                # Find all nonzero indices in roi_points
                nonzero_indices = torch.all(roi_points != 0, dim=1)
                if nonzero_indices.sum() == 0:
                    continue
                roi_points = roi_points[nonzero_indices]
                roi_features = roi_features[nonzero_indices]

                min_bound = roi_points.min(dim=0).values
                max_bound = roi_points.max(dim=0).values
                cube_center = (min_bound + max_bound) / 2
                cube_size = torch.max(max_bound - min_bound)

                roi_points = (roi_points - cube_center) / cube_size + 0.5
                # Mask out all points outside [0, 1]
                pixelate_mask = ((roi_points >= 0) & (roi_points <= 1)).all(dim=1)
                if pixelate_mask.sum() == 0:
                    continue

                roi_points = roi_points[pixelate_mask]
                roi_features = roi_features[pixelate_mask]
                # segment_points = (segment_points - min_bound) / cube_size

                segment_grid = pixelate_point_clouds(
                    roi_points, roi_features, self.triplane_dims
                )

                result_feature_2d.append(roi_features_2d)
                result_grid.append(segment_grid)
                valid_indices.append(idx)

            result_feature_2d = (
                torch.stack(result_feature_2d)
                if len(result_feature_2d) > 0
                else torch.zeros(0)
            )
            result_grid = (
                torch.stack(result_grid) if len(result_grid) > 0 else torch.zeros(0)
            )
            batched_triplanes.append(
                {
                    "features_2d": result_feature_2d,
                    "features": result_grid,
                    "indices": (
                        torch.LongTensor(valid_indices)
                        if len(valid_indices) > 0
                        else []
                    ),
                }
            )
        return batched_triplanes


class HighResCubeForwardProjection(CubeForwardProjection):
    def forward(
        self,
        encoder_features,
        depths,
        intrinsics,
        masks,
        enable_projection=None,
        extrude_depth=None,
    ):
        """
        Args:
            encoder_features: (N, C, H, W) tensor of encoder features

        Returns:
            batched_triplanes: list of dictionaries containing: triplane features, cube center, cube size
        """
        batched_triplanes = []
        for batch_id in range(len(depths)):
            encoder_feature = encoder_features[batch_id]  # C, H, W: 256, 128, 168
            C, H, W = encoder_feature.shape
            depth = depths[batch_id]  # H, W
            intrinsic = copy.deepcopy(intrinsics[batch_id])  # 3, 3
            segments = masks[batch_id]  # List of masks
            if len(segments) == 0:
                batched_triplanes.append({})
                continue

            intrinsic[2, 2] *= 4
            # Get camera coords
            intrinsic_inverse = torch.inverse(
                intrinsic
            )  # The divisibility padding is applied to down right, so the intrinsic wouldn't change

            # feature_depth = F.interpolate(
            #     depth[None, None], size=encoder_feature.shape[-2:], mode="nearest"
            # ).view(
            #     *encoder_feature.shape[-2:]
            # )  # Align depth with features
            feature_depth = depth[::4, ::4]  # Downsample
            encoder_feature = F.interpolate(
                encoder_feature.unsqueeze(0),
                size=(feature_depth.shape[0], feature_depth.shape[1]),
                mode="bilinear",
            ).squeeze(0)
            feature_points = project_depth_to_camera_coordinates(
                feature_depth, intrinsic_inverse
            )

            result_grid = []
            valid_indices = []
            result_feature_2d = []

            for idx, mask in enumerate(segments):
                if (enable_projection is not None) and (
                    not enable_projection[batch_id][idx]
                ):
                    continue
                segment_mask = mask[::4, ::4]  # Downsample to feature map size
                segment_mask = segment_mask & (feature_depth > 0)
                if segment_mask.sum() == 0:
                    continue
                segment_points = feature_points[segment_mask].view(-1, 3)
                segment_features = encoder_feature.permute(1, 2, 0)[segment_mask].view(
                    -1, C
                )

                # extrude_depth: float, add [-extrude_depth, extrude_depth] z-offset to segment_points, repeat segment_features (2 * num_repetition + 1) times
                if extrude_depth is not None:
                    num_repetition = 3
                    z_offsets = torch.linspace(
                        -extrude_depth,
                        extrude_depth,
                        2 * num_repetition + 1,
                        device=segment_points.device,
                    )
                    segment_points_repeated = segment_points.unsqueeze(0).repeat(
                        len(z_offsets), 1, 1
                    )
                    segment_points_repeated[..., 2] += z_offsets.view(-1, 1)
                    segment_points = segment_points_repeated.reshape(-1, 3)
                    segment_features = (
                        segment_features.unsqueeze(0)
                        .repeat(len(z_offsets), 1, 1)
                        .reshape(-1, segment_features.shape[-1])
                    )

                min_bound = segment_points.min(dim=0).values
                max_bound = segment_points.max(dim=0).values
                cube_center = (min_bound + max_bound) / 2
                cube_size = torch.max(max_bound - min_bound)

                segment_points = (segment_points - cube_center) / cube_size + 0.5
                # Mask out all points outside [0, 1]
                pixelate_mask = ((segment_points >= 0) & (segment_points <= 1)).all(
                    dim=1
                )
                if pixelate_mask.sum() == 0:
                    continue
                segment_points = segment_points[pixelate_mask]
                segment_features = segment_features[pixelate_mask]
                # segment_points = (segment_points - min_bound) / cube_size

                segment_grid = pixelate_point_clouds(
                    segment_points, segment_features, self.triplane_dims
                )

                result_grid.append(segment_grid)
                valid_indices.append(idx)

                masked_features = encoder_feature.clone().detach()
                masked_features[:, ~segment_mask] = 0
                roi_features_2d = crop_roi(
                    masked_features, segment_mask, self.triplane_dims
                )  # 256, 32, 32
                result_feature_2d.append(roi_features_2d)

            result_grid = (
                torch.stack(result_grid) if len(result_grid) > 0 else torch.zeros(0)
            )
            result_feature_2d = (
                torch.stack(result_feature_2d)
                if len(result_feature_2d) > 0
                else torch.zeros(0)
            )
            batched_triplanes.append(
                {
                    "features_2d": result_feature_2d,
                    "features": result_grid,
                    "indices": (
                        torch.LongTensor(valid_indices)
                        if len(valid_indices) > 0
                        else []
                    ),
                }
            )
        return batched_triplanes


class ImagePlaneProjection(nn.Module):
    @configurable
    def __init__(self, triplane_dims=32):
        super().__init__()
        self.triplane_dims = triplane_dims

    @classmethod
    def from_config(cls, cfg):
        return {
            "triplane_dims": cfg.MODEL.TRIPLANE_VAE.DIMENSIONS,
        }

    def crop_2d_feature(self, feature, mask):
        mask_indices = torch.nonzero(mask, as_tuple=False)
        x_1, y_1 = mask_indices.min(dim=0)[0]
        x_2, y_2 = mask_indices.max(dim=0)[0]
        boxes = [torch.tensor([[y_1, x_1, y_2, x_2]]).float().cuda()]
        cropped_feature = torchvision.ops.roi_align(
            feature.unsqueeze(0).float(), boxes, self.triplane_dims
        )[0]
        return cropped_feature

    def forward(
        self,
        encoder_features,
        depths,
        intrinsics,
        masks,
        enable_projection=None,
        extrude_depth=None,
    ):
        """
        Args:
            encoder_features: (N, C, H, W) tensor of encoder features

        Returns:
            batched_triplanes: list of dictionaries containing: triplane features, cube center, cube size
        """
        if extrude_depth is not None:
            raise NotImplementedError
        batched_triplanes = []
        for batch_id in range(len(masks)):
            encoder_feature = encoder_features[batch_id]  # C, H, W: 256, 128, 168
            segments = masks[batch_id]  # List of masks
            if len(segments) == 0:
                batched_triplanes.append({})
                continue

            result_features = []
            valid_indices = []
            for idx, mask in enumerate(segments):
                if (enable_projection is not None) and (
                    not enable_projection[batch_id][idx]
                ):
                    continue
                segment_mask = mask[::4, ::4]  # Downsample to feature map size
                segment_mask = segment_mask
                if segment_mask.sum() < 10:
                    continue

                segment_feature = crop_roi(
                    encoder_feature.clone().detach(), segment_mask, self.triplane_dims
                )
                if segment_feature is None:
                    continue

                result_features.append(segment_feature)
                valid_indices.append(idx)
            result_features = (
                torch.stack(result_features)
                if len(result_features) > 0
                else torch.zeros(0)
            )
            batched_triplanes.append(
                {
                    "features": result_features,
                    "features_2d": result_features,
                    "indices": (
                        torch.LongTensor(valid_indices)
                        if len(valid_indices) > 0
                        else []
                    ),
                }
            )
        return batched_triplanes


def project_depth_to_camera_coordinates(
    depth: torch.Tensor, intrinsic_inverse: torch.Tensor
) -> torch.Tensor:
    """
    Project the depth map to camera coordinates
    Args:
        depth (torch.Tensor): (H, W) tensor of depth values
        intrinsic_inverse (torch.Tensor): (3, 3) tensor of the inverse intrinsic matrix

    Returns:
        torch.Tensor: (H, W, 3) tensor of camera coordinates
    """
    pos_pix = torch.meshgrid(
        torch.arange(depth.shape[-1], device=depth.device),  # 648
        torch.arange(depth.shape[-2], device=depth.device),  # 484
        indexing="ij",
    )
    pos_pix = torch.stack(pos_pix, dim=-1).float()  # (W, H, 2)
    pos_pix = torch.cat([pos_pix, torch.ones_like(pos_pix[..., 0:1])], dim=-1).to(
        intrinsic_inverse
    )  # (W, H, 3)
    pos_cam = torch.mm(pos_pix.view(-1, 3), intrinsic_inverse.T)  # (W*H, 3)
    pos_cam = pos_cam.view(depth.shape[-1], depth.shape[-2], 3).transpose(
        0, 1
    )  # (H, W, 3)
    pos_cam = pos_cam / pos_cam[..., 2:3]  # Normalize by depth
    pos_cam = pos_cam * depth.unsqueeze(-1)  # (H, W, 3)
    return pos_cam


def rasterize_point_clouds(
    points: torch.Tensor,
    features: torch.Tensor,
    grid_size: int,
    plane_axis: tuple[int, int],
):
    """
    Rasterize a point cloud into a grid with averaged features.
    The outlier points are ignored.
    Args:
        points (torch.Tensor): (N, 3) tensor of point cloud coordinates, in the range [0, 1].
        features (torch.Tensor): (N, C) tensor of point cloud features.
        grid_size (int): size of the grid.
        plane_axis (tuple[int, int]): axis of the plane to rasterize onto, e.g. (0, 1) for XY plane.
    Returns:
        torch.Tensor: (grid_size, grid_size, C) tensor of averaged
    """
    feature_dim = features.size(1)

    # Scale to grid size
    grid_x = (points[:, plane_axis[0]] * (grid_size - 1)).long()
    grid_y = (points[:, plane_axis[1]] * (grid_size - 1)).long()

    # Initialize the grid with zeros
    grid = torch.zeros(
        (grid_size, grid_size, feature_dim),
        dtype=features.dtype,
        device=features.device,
    )

    # Use scatter_add to accumulate colors in cells
    indices = grid_x * grid_size + grid_y
    try:
        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
    except Exception as e:
        import pdb

        pdb.set_trace()

    # Accumulate colors for each unique grid cell
    accumulated_features = torch.zeros(
        (unique_indices.size(0), feature_dim),
        dtype=features.dtype,
        device=features.device,
    )
    accumulated_features.scatter_add_(
        0, inverse_indices.unsqueeze(1).expand(-1, 3), features
    )

    # Count number of points per grid cell for averaging
    counts = torch.zeros(
        unique_indices.size(0), dtype=torch.float32, device=features.device
    )
    counts.scatter_add_(
        0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32)
    )

    # Average the colors in each grid cell
    averaged_features = accumulated_features / counts.unsqueeze(1)

    # Map averaged colors back to the grid
    grid_x_unique = unique_indices // grid_size
    grid_y_unique = unique_indices % grid_size
    grid[grid_x_unique, grid_y_unique] = averaged_features

    return grid


def pixelate_point_clouds(
    points: torch.Tensor,
    features: torch.Tensor,
    grid_size: int,
):
    """
    Pixelate a point cloud into a grid with averaged features.
    The outlier points are ignored.
    Args:
        points (torch.Tensor): (N, 3) tensor of point cloud coordinates, in the range [0, 1].
        features (torch.Tensor): (N, C) tensor of point cloud features.
        grid_size (int): size of the grid.
    Returns:
        torch.Tensor: (grid_size, grid_size, grid_size, C) tensor of averaged
    """
    feature_dim = features.size(1)

    # Quantize coordinates to grid indices
    grid_indices = (points * (grid_size - 1)).long()

    # Initialize grid for storing colors and count
    grid_features = torch.zeros(
        (grid_size, grid_size, grid_size, feature_dim),
        dtype=features.dtype,
        device=features.device,
    )
    count_grid = torch.zeros(
        (grid_size, grid_size, grid_size), dtype=torch.float32, device=features.device
    )

    # Flatten grid indices for easy indexing
    flattened_indices = (
        grid_indices[:, 0] * grid_size * grid_size
        + grid_indices[:, 1] * grid_size
        + grid_indices[:, 2]
    )

    # Aggregate colors and counts using scatter_add
    grid_features = grid_features.view(-1, feature_dim)
    count_grid = count_grid.view(-1)

    grid_features.index_add_(0, flattened_indices, features)
    count_grid.index_add_(
        0,
        flattened_indices,
        torch.ones_like(flattened_indices, dtype=torch.float32, device=features.device),
    )

    # Reshape back to grid shape
    grid_features = grid_features.view(grid_size, grid_size, grid_size, feature_dim)
    count_grid = count_grid.view(grid_size, grid_size, grid_size)

    # Avoid division by zero
    count_grid[count_grid == 0] = 1

    # Average colors in each voxel
    grid_features /= count_grid.unsqueeze(-1)
    return grid_features
