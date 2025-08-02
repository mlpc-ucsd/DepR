"""
Example usage:

python -m scripts.eval_scene --gt-pcd-dir output/gt/pcds --pred-dir output/baselines/evaluation/preds --save-dir output/baselines/evaluation/results --method uni3d

"""

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import jsonlines
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
import open3d as o3d
from detectron2.utils.logger import setup_logger
from accelerate import Accelerator
from depr import utils


def get_depr_mesh(pred_dir, uid):
    uid, sub = uid.split("_")
    return pred_dir / uid / "mesh_scene" / f"{sub}.glb"


def get_uni3d_mesh(pred_dir, uid):
    return (pred_dir / str(uid) / "mesh_geometry.ply").as_posix()


def get_buol_mesh(pred_dir, uid):
    return (pred_dir / f"{uid}.ply").as_posix()


def get_gen3dsr_mesh(pred_dir, uid):
    return (pred_dir / str(uid) / "reconstruction/full_scene.glb").as_posix()


def get_dpa_mesh(pred_dir, uid):
    return (pred_dir / str(uid) / "merged_mesh.obj").as_posix()


def get_instpifu_mesh(pred_dir, uid):
    return (pred_dir / f"rendertask{uid}" / "scene.glb").as_posix()


@torch.no_grad()
def evaluate_single(logger, gt_pcd_dir, pred_dir, save_dir, uid, mesh_getter):
    aligned_pcd_path = save_dir / f"{uid}.npz"
    if "_" in uid:
        real_uid = uid.split("_")[0]
    else:
        real_uid = uid
    gt_pts = np.load(gt_pcd_dir / f"{real_uid}.npz")["pcds"]
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)
    gt_pts = torch.from_numpy(gt_pts).float()

    if not aligned_pcd_path.exists():
        pred_mesh_path = mesh_getter(pred_dir, uid)
        pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)
        try:
            pred_pcd = pred_mesh.sample_points_uniformly(10000)
        except:
            logger.warning(
                f"Error sampling points from mesh {pred_mesh_path}. Skipping {uid}."
            )
            return
        pred_pcd = pred_pcd.scale(
            (
                gt_pcd.get_minimal_oriented_bounding_box().volume()
                / pred_pcd.get_minimal_oriented_bounding_box().volume()
            )
            ** (1 / 3),
            center=np.array((0.0, 0.0, 0.0)),
        )
        pred_pcd = pred_pcd.translate(gt_pcd.get_center() - pred_pcd.get_center())
        pred_pts = utils.geom.o3d_pcd_to_tensor(pred_pcd)
        transform = utils.sample.get_object_transformations(
            [pred_pts],
            [gt_pts],
            num_steps=200,
            verbose=False,
        ).squeeze(0)
        aligned_pts = utils.transforms.apply_transformation_matrix(pred_pts, transform)
        np.savez(aligned_pcd_path, pcds=aligned_pts.numpy())
    else:
        aligned_pts = np.load(aligned_pcd_path)["pcds"]
        aligned_pts = torch.from_numpy(aligned_pts).float()

    aligned_pts_cuda = aligned_pts.unsqueeze(0).cuda()
    gt_pts_cuda = gt_pts.unsqueeze(0).cuda()
    cd = chamfer_distance(aligned_pts_cuda, gt_pts_cuda)[0].item()
    cd_s = chamfer_distance(aligned_pts_cuda, gt_pts_cuda, single_directional=True)[
        0
    ].item()
    f_score = utils.metrics.f_score(
        aligned_pts.numpy(), gt_pts.numpy(), tau=0.002
    )  # InstPIFu
    f_score_2 = utils.metrics.f_score(
        aligned_pts.numpy(), gt_pts.numpy(), tau=0.1
    )  # DeepPriorAssembly
    result_dict = {
        "cd": cd,
        "cd_s": cd_s,
        "f_score": f_score,
        "f_score_2": f_score_2,
    }
    with open(save_dir / f"{uid}.json", "w") as f:
        json.dump(result_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=str, default="datasets/front3d_pifu/meta/test_scene.jsonl"
    )
    parser.add_argument("--gt-pcd-dir", type=str, required=True)
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["gen3dsr", "dpa", "buol", "instpifu", "uni3d", "depr"],
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    match args.method:
        case "gen3dsr":
            mesh_getter = get_gen3dsr_mesh
        case "dpa":
            mesh_getter = get_dpa_mesh
        case "buol":
            mesh_getter = get_buol_mesh
        case "instpifu":
            mesh_getter = get_instpifu_mesh
        case "uni3d":
            mesh_getter = get_uni3d_mesh
        case "depr":
            mesh_getter = get_depr_mesh

    with jsonlines.open(args.metadata, "r") as reader:
        all_uids = [line["image_id"] for line in reader]

    accelerator = Accelerator()

    if args.method == "depr":
        all_uids = [f"{uid}_{i}" for uid in all_uids for i in range(args.num_samples)]

    sharded_uids = all_uids[accelerator.process_index :: accelerator.num_processes]

    save_dir = Path(args.save_dir) / args.method
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        output=str(save_dir / "eval.log"),
        color=False,
    )

    pred_dir = Path(args.pred_dir)
    if args.method != "depr":
        pred_dir = pred_dir / args.method

    for uid in tqdm(sharded_uids):
        evaluate_single(
            logger,
            Path(args.gt_pcd_dir),
            pred_dir,
            save_dir,
            uid,
            mesh_getter,
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Collate results
        all_cds = []
        all_cd_s = []
        all_f_scores = []
        all_f_scores_2 = []
        for uid in all_uids:
            record_path = save_dir / f"{uid}.json"
            if not record_path.exists():
                logger.warning(f"Skipping {uid}, no record found.")
                continue
            with record_path.open() as f:
                result_dict = json.load(f)
            all_cds.append(result_dict["cd"])
            all_cd_s.append(result_dict["cd_s"])
            all_f_scores.append(result_dict["f_score"])
            all_f_scores_2.append(result_dict["f_score_2"])

        logger.info("======== Summary ========")
        logger.info(f"Method: {args.method}")
        logger.info(f"Number of samples: {len(all_cds)}")
        logger.info(f"CD: {np.mean(all_cds)}")
        logger.info(f"CD-S: {np.mean(all_cd_s)}")
        logger.info(f"F-Score (tau=0.002): {np.mean(all_f_scores)}")
        logger.info(f"F-Score (tau=0.1): {np.mean(all_f_scores_2)}")
        logger.info("==========================")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
