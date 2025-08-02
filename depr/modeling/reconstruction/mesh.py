import numpy as np
import skimage.measure
import torch
import trimesh
import open3d as o3d
from typing import Union


def create_mesh(
    model,
    shape_feature,
    N=256,
    max_batch=1000000,
    level_set=0.0,
    occupancy=False,
    point_cloud=None,
    from_plane_features=False,
    from_pc_features=False,
    alpha=1.0,
    voxel_origin=np.array([-1, -1, -1], dtype=np.float32),
    cube_size=2.0,
    return_type="trimesh",
) -> Union[trimesh.Trimesh, o3d.geometry.TriangleMesh]:

    model.eval()
    device = next(model.parameters()).device

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_size = cube_size / (N - 1)
    cube = create_cube(N)
    cube_points = cube.shape[0]
    head = 0
    while head < cube_points:
        query = cube[head : min(head + max_batch, cube_points), 0:3].unsqueeze(0)
        if from_plane_features:
            pred_sdf = (
                model.forward_with_plane_features(
                    shape_feature.to(device), query.to(device)
                )
                .detach()
                .cpu()
            )
        else:
            pred_sdf = model(shape_feature.to(device), query.to(device)).detach().cpu()
        cube[head : min(head + max_batch, cube_points), 3] = pred_sdf.squeeze()
        head += max_batch
    # for occupancy instead of SDF, subtract 0.5 so the surface boundary becomes 0
    sdf_values = cube[:, 3] - 0.5 if occupancy else cube[:, 3]
    sdf_values *= alpha
    sdf_values = sdf_values.reshape(N, N, N)
    return convert_sdf_samples_to_mesh(
        sdf_values, voxel_origin, voxel_size, return_type, level_set
    )


def create_cube(N):
    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples


def convert_sdf_samples_to_mesh(
    pytorch_3d_sdf_tensor: torch.Tensor,
    voxel_grid_origin: np.ndarray,
    voxel_size: float,
    return_type: str,
    level_set=0.0,
) -> Union[trimesh.Trimesh, o3d.geometry.TriangleMesh]:
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a numpy array of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    # use marching_cubes_lewiner or marching_cubes depending on pytorch version
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level_set, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print("Marching cubes failed")
        return None

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = verts + voxel_grid_origin.reshape(1, 3)
    if return_type == "trimesh":
        return trimesh.Trimesh(mesh_points, faces)
    elif return_type == "open3d":
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_points), o3d.utility.Vector3iVector(faces)
        )
        return mesh
    else:
        raise ValueError("Unknown return_type: %s" % return_type)
