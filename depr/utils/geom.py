from typing import List
import trimesh
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from depr.modeling.reconstruction import project_depth_to_camera_coordinates
from depr.modeling.reconstruction.mesh import create_mesh


def get_o3d_point_cloud(points: np.ndarray):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def o3d_pcd_to_tensor(
    pcd: o3d.geometry.PointCloud, num_sample_points=None
) -> torch.Tensor:
    pts = np.asarray(pcd.points)
    if num_sample_points is not None:
        random_inds = np.random.choice(pts.shape[0], num_sample_points)
        pts = pts[random_inds]
    return torch.from_numpy(pts).float()


def sample_points_from_o3d_mesh(mesh, num_points, sample_method="uniform"):
    if sample_method not in ["uniform", "poisson"]:
        raise ValueError(f"Unknown sampling method: {sample_method}")
    if sample_method == "uniform":
        points = mesh.sample_points_uniformly(num_points).points
    elif sample_method == "poisson":
        points = mesh.sample_points_poisson_disk(num_points).points
    return torch.from_numpy(np.asarray(points)).float()


def get_normalized_pcd(pcd: torch.Tensor):
    # Normalize via bounding box
    vmax = pcd.max(dim=-2, keepdim=True).values
    vmin = pcd.min(dim=-2, keepdim=True).values
    cube_size = (vmax - vmin).max(dim=-1, keepdim=True).values
    cube_center = (vmin + vmax) / 2
    pcd = (pcd - cube_center) / cube_size
    return pcd


def get_normalized_pcd_with_centroid(pcd: torch.Tensor, return_extra=False):
    # Normalize via centroid to unit sphere
    centroid = pcd.mean(dim=-2, keepdim=True)
    centered_pcd = pcd - centroid
    scale = centered_pcd.norm(dim=-1, keepdim=True).max(dim=-2, keepdim=True).values
    unit_pcd = centered_pcd / scale
    if return_extra:
        return unit_pcd, centroid.squeeze(-2), scale.squeeze(-2, -1)
    else:
        return unit_pcd


def triplanes_to_scene(
    model, triplanes, transformations=None, resolution=64, include_axis=True
):
    final_scene = trimesh.Scene()
    if include_axis:
        final_scene.add_geometry(trimesh.creation.axis(axis_length=0.1))
    for idx, triplane in enumerate(triplanes):
        mesh_pred = create_mesh(
            model.sdf_model,
            triplane.unsqueeze(0),
            N=resolution,
            max_batch=2**21,
            from_plane_features=True,
            alpha=1,
        )
        if transformations is not None:
            if transformations[idx] is not None:
                mesh_pred.apply_transform(transformations[idx])
                final_scene.add_geometry(mesh_pred)
        else:
            final_scene.add_geometry(mesh_pred)
    return final_scene


def triplanes_to_point_clouds(
    model: nn.Module,
    denormalized_triplanes: torch.Tensor,
    num_points=5000,
    resolution=64,
    sample_method="poisson",
    return_meshes=False,
) -> List[o3d.geometry.PointCloud]:
    source_list = []
    all_meshes = []
    for triplane in denormalized_triplanes:
        source_mesh = create_mesh(
            model.sdf_model,
            triplane.unsqueeze(0),
            N=resolution,
            max_batch=2**21,
            from_plane_features=True,
            alpha=1,
            return_type="open3d",
        )
        if source_mesh is None:
            print("Failed to create mesh")
            continue
        source_pts = sample_points_from_o3d_mesh(source_mesh, num_points, sample_method)
        source_list.append(source_pts)
        all_meshes.append(source_mesh)
    if return_meshes:
        return source_list, all_meshes
    return source_list


def save_colored_pointcloud(pointclouds, filename: str):
    """
    Save an (N, K, 3) pointcloud with each instance colored differently to a PLY file.

    Parameters:
        pointclouds (np.ndarray): The point cloud of shape (N, K, 3).
        filename (str): Output filename, e.g., 'colored_cloud.ply'.
    """
    N, K, _ = pointclouds.shape
    all_points = pointclouds.reshape(-1, 3)

    colors = np.random.rand(N, 3)
    all_colors = np.repeat(colors, K, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.io.write_point_cloud(filename, pcd)


def save_scene_meshes(meshes: List[o3d.geometry.TriangleMesh], filename):
    scene_mesh = trimesh.Scene()
    for idx, mesh in enumerate(meshes):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        geometry = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        scene_mesh.add_geometry(geometry, geom_name=f"mesh_{idx}")
    scene_mesh.export(filename)


def retain_valid_clusters(
    pcd,
    eps=0.1,
    min_points=100,
    min_cluster_size_ratio=0.15,
    print_progress=True,
):
    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=print_progress
        )
    )
    num_clusters = labels.max() + 1
    if print_progress:
        print(f"Number of clusters: {num_clusters}")

    cluster_sizes = np.bincount(labels[labels >= 0])
    cluster_ratio = cluster_sizes / cluster_sizes.sum()
    kept_cluster = np.where(cluster_ratio > min_cluster_size_ratio)[0]
    inliers = np.isin(labels, kept_cluster)
    pcd_clean = pcd.select_by_index(np.where(inliers)[0])
    return pcd_clean


def prepare_depth_point_cloud(
    depth,
    masks,
    intrinsics,
    num_sample_points,
    clean_up_with_cluster=True,
):
    intrinsic_inverse = torch.inverse(intrinsics)
    projected_points = project_depth_to_camera_coordinates(depth, intrinsic_inverse)
    target_list = []
    for segment_mask in masks:
        valid_mask = segment_mask & (depth > 0)
        segment_points = projected_points[valid_mask].view(-1, 3)
        o3d_pcd = get_o3d_point_cloud(segment_points.cpu().numpy())
        o3d_pcd, _ = o3d_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if clean_up_with_cluster:
            o3d_pcd = retain_valid_clusters(o3d_pcd)
        target_points = o3d_pcd_to_tensor(o3d_pcd, num_sample_points=num_sample_points)
        target_list.append(target_points)
    return target_list
