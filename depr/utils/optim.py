import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.loss import chamfer_distance
from .geom import get_o3d_point_cloud


def augment_transformation_matrix(matrix):
    size = matrix.shape[-1]
    augmented_matrix = matrix.new_zeros(*matrix.shape[:-2], size + 1, size + 1)
    augmented_matrix[..., :size, :size] = matrix
    augmented_matrix[..., -1, -1] = 1.0
    return augmented_matrix


def random_rotation_matrix_y(num_repeats, num_instances, device):
    rot_y = (
        torch.rand((num_repeats, num_instances), device=device, dtype=torch.float32)
        * 2
        * torch.pi
    )
    rot = rot_y.new_zeros(num_repeats, num_instances, 3)
    rot[..., 1] = rot_y
    return euler_angles_to_matrix(rot, "XYZ")


class PoseEstimator(nn.Module):
    def __init__(self, num_repeats, num_instances, dof=7, enable_global_rotation=False):
        super().__init__()
        assert dof in (5, 7), "DoF must be either 5 or 7"

        if dof == 5:
            self.rotation_y = nn.Parameter(
                torch.zeros(num_repeats, num_instances), requires_grad=True
            )
        else:
            self.rotation = nn.Parameter(
                torch.zeros(num_repeats, num_instances, 3), requires_grad=True
            )
        self.translation = nn.Parameter(
            torch.zeros(num_repeats, num_instances, 3), requires_grad=True
        )
        self.scale = nn.Parameter(
            torch.ones(num_repeats, num_instances), requires_grad=True
        )
        if enable_global_rotation:
            self.global_rotation = nn.Parameter(
                torch.zeros(num_repeats, 3), requires_grad=True
            )
        else:
            self.global_rotation = None
        self.dof = dof
        self.num_repeats = num_repeats
        self.num_instances = num_instances

    @property
    def eulers(self):
        if self.dof == 5:
            angles = self.rotation_y.new_zeros(self.num_repeats, self.num_instances, 3)
            angles[..., 1] = self.rotation_y
            return angles
        else:
            return self.rotation

    @torch.no_grad()
    def get_transformation_matrix(self):
        return self._get_transformation_matrix()
    
    def _get_transformation_matrix(self):
        result = euler_angles_to_matrix(self.eulers, "XYZ")
        result = augment_transformation_matrix(result)
        result[..., :3, 3] = self.translation
        result[..., :3, :] *= self.scale[:, :, None, None]
        if self.global_rotation is not None:
            global_rot_matrix = euler_angles_to_matrix(self.global_rotation, "XYZ")
            global_rot_matrix = augment_transformation_matrix(global_rot_matrix)
            global_rot_matrix = global_rot_matrix[:, None, :, :]
            result = torch.matmul(global_rot_matrix, result)
        return result

    def get_cd_loss(self, transformed_points, target_points, single_directional=False):
        # We may want to use single-directional if target_points are from partial shape
        point_dim = transformed_points.shape[-1]
        if target_points.ndim == 3:
            target_points = target_points[None].repeat(self.num_repeats, 1, 1, 1)
        num_points = transformed_points.shape[-2]
        cd_loss = chamfer_distance(
            target_points.reshape(-1, num_points, point_dim),
            transformed_points.reshape(-1, num_points, point_dim),
            batch_reduction=None,
            point_reduction="mean",
            single_directional=single_directional,
        )[0]
        cd_loss = cd_loss.reshape(self.num_repeats, self.num_instances)
        return cd_loss

    def forward(self, source_points, target_points=None):
        # Num repeats: B
        # Source / Target points: [N (num_insts), K (num_points), 3]
        # [N, K, 3] x [B, N, 3, 3] -> [B, N, K, 3] + [B, N, 1, 3]
        augmented_points = torch.cat([source_points, torch.ones_like(source_points[..., :1])], dim=-1)        
        transform = self._get_transformation_matrix()
        transformed_points = torch.matmul(augmented_points, transform.transpose(-1, -2))
        transformed_points = transformed_points[..., :3]
        if target_points is None:
            return transformed_points
        return self.get_cd_loss(transformed_points, target_points)


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 0.5
    print(
        ":: Apply fast global registration with distance threshold %.3f"
        % distance_threshold
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def open3d_global_registration(source_pts, target_pts, verbose=False):
    num_instances = source_pts.shape[0]
    voxel_size = 0.05

    optimized_transforms = []
    for idx in range(num_instances):
        source_pcd = get_o3d_point_cloud(source_pts[idx])
        target_pcd = get_o3d_point_cloud(target_pts[idx])
        source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
        result_fast = execute_fast_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
        if verbose:
            print(f"Instance {idx}: Registration result: {result_fast}")
        transforms = np.asarray(result_fast.transformation)
        optimized_transforms.append(transforms)

    optimized_transforms = np.stack(optimized_transforms, axis=0)
    optimized_transforms = torch.as_tensor(optimized_transforms, dtype=torch.float32)
    return optimized_transforms
