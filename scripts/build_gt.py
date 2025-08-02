import argparse
import numpy as np
import jsonlines
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from depr import utils


def build_gt_point_cloud(output_dir, uid, num_samples):
    gt_scene = utils.data.build_gt_scene(uid)
    if len(gt_scene) == 0:
        return
    mesh_out = output_dir / "meshes"
    pcd_out = output_dir / "pcds"
    mesh_out.mkdir(parents=True, exist_ok=True)
    pcd_out.mkdir(parents=True, exist_ok=True)
    utils.geom.save_scene_meshes(gt_scene, mesh_out / f"{uid}.glb")
    scene_mesh = o3d.geometry.TriangleMesh()
    for mesh in gt_scene:
        scene_mesh += mesh
    pcds = scene_mesh.sample_points_uniformly(num_samples)
    pcds = np.asarray(pcds.points)
    np.savez(pcd_out / f"{uid}.npz", pcds=pcds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=str, default="datasets/front3d_pifu/meta/test_scene.jsonl"
    )
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    with jsonlines.open(args.metadata, "r") as reader:
        all_uids = [line["image_id"] for line in reader]

    for uid in tqdm(all_uids):
        build_gt_point_cloud(Path(args.out_dir), uid, num_samples=10000)


if __name__ == "__main__":
    main()
