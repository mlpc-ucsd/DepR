import logging
import pickle
import json
import easydict
import numpy as np
from PIL import Image
import torch
from pytorch3d.loss import chamfer_distance
import open3d as o3d
from safetensors.torch import load_file as load_safetensors
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.projects.deeplab import add_deeplab_config
from depr import add_model_2d_config, add_depr_config, utils
from depr.modeling.reconstruction import mesh

logger = logging.getLogger(__name__)


def _preprocess_depth(pickled_data, depth):
    gt_depth = pickled_data["depth_map"]
    gt_depth = (1 - gt_depth / 255) * 10
    gt_depth = torch.from_numpy(gt_depth).float()

    if depth is None:
        depth = gt_depth
    else:
        aligned = utils.transforms.align_depth(depth, gt_depth, mask=gt_depth > 0)
        depth = torch.from_numpy(aligned).float()

    return depth


def _read_pred_meshes(model, mesh_dir, pred_embeds=None, pred_embed_path=None):
    has_pred_mesh = mesh_dir.exists() and len(list(mesh_dir.glob("*.ply"))) > 0
    if pred_embeds is None:
        pred_embeds = torch.from_numpy(np.load(pred_embed_path)).cuda()
    num_insts = len(pred_embeds)
    if has_pred_mesh:
        pred_meshes = []
        for idx in range(num_insts):
            o3d_mesh = o3d.io.read_triangle_mesh(str(mesh_dir / f"{idx}.ply"))
            if not o3d_mesh.is_empty():
                pred_meshes.append(o3d_mesh)
            else:
                pred_meshes.append(None)
    else:
        pred_triplane = model.triplane_vae.decode(pred_embeds)
        triplanes = model.denormalize_triplanes(pred_triplane)
        pred_meshes = []
        for idx in range(len(triplanes)):
            o3d_mesh = mesh.create_mesh(
                model.sdf_model,
                triplanes[idx].unsqueeze(0),
                N=128,
                max_batch=2**21,
                from_plane_features=True,
                alpha=1,
                return_type="open3d",
            )
            pred_meshes.append(o3d_mesh)
    return pred_meshes


def _prepare_pred_meshes(model, output_path, sample_id, use_guiding):
    pred_path = output_path / f"embed_{sample_id}.npy"
    pred_meshes = _read_pred_meshes(
        model,
        output_path / f"mesh_{sample_id}",
        pred_embeds=None,
        pred_embed_path=pred_path,
    )
    if use_guiding:
        # select based on guided results
        guided_embed_path = output_path / f"guided_embed_{sample_id}.npz"
        guided_transform_path = output_path / f"guided_transforms_{sample_id}.npz"
        guided_info = np.load(guided_embed_path)
        guided_losses = guided_info["guided_losses"]
        # if guided_transform_path.exists():
        #     guided_transform_info = np.load(guided_transform_path)
        #     guided_transform_losses = guided_transform_info["losses"]
        guided_meshes = _read_pred_meshes(
            model,
            output_path / f"guided_mesh_{sample_id}",
            pred_embeds=torch.from_numpy(guided_info["guided_embeds"]).cuda(),
            pred_embed_path=None,
        )
        is_valid_guidance = guided_losses < 5
        pred_meshes = np.where(is_valid_guidance, guided_meshes, pred_meshes)
    return pred_meshes


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_model_2d_config(cfg)
    add_depr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT_DIR = args.output
    cfg.MODEL.WEIGHTS = args.model
    cfg.SEED = 0
    cfg.freeze()
    return cfg


def reconstruct_object(
    model,
    output_dir,
    data_id,
    depth=None,
    use_guiding=False,
    use_gt_transform=False,
    use_sam=False,
    use_gt_segm=False,
    inf_sample_num=3,
    override=False,
    save_rgb=False,
    save_depth=False,
    save_mesh=False,
):
    assert not (
        use_sam and use_gt_transform
    ), "Cannot use both SAM and GT transform at the same time"
    assert not (
        use_sam and use_gt_segm
    ), "Cannot use both SAM and GT segmentation at the same time"

    device = torch.device("cuda")

    with open(
        f"datasets/front3d_pifu/data/pickled_data/test/rendertask{data_id}.pkl",
        "rb",
    ) as f:
        pickled_data = pickle.load(f)
    output_path = output_dir / f"{data_id}"
    output_path.mkdir(parents=False, exist_ok=True)

    pred_paths = list(output_path.glob("embed_*.npy"))
    guided_pred_path = list(output_path.glob("guided_embed_*.npz"))
    if not override:
        if not use_guiding and len(pred_paths) >= inf_sample_num:
            return
        if use_guiding and len(guided_pred_path) >= inf_sample_num:
            return

    logger.info(f"Processing {data_id}")
    image = torch.as_tensor(pickled_data["rgb_img"]).permute(2, 0, 1)  # 3, 484, 646
    cam_info = pickled_data["camera"]
    scale = cam_info["scale_factor"]
    K = np.asarray(cam_info["K"], dtype=np.float32)
    K[0] /= scale
    K[1] /= scale
    K[2, 2] = 1
    intrinsics = torch.as_tensor(K)
    wrd2cam_matrix = np.asarray(cam_info["wrd2cam_matrix"], dtype=np.float32)

    depth = _preprocess_depth(pickled_data, depth)

    if use_sam:
        masks, inst_ids, labels = utils.data.read_sam_segmentation(
            data_id, return_labels=True
        )
    elif use_gt_segm:
        masks, inst_ids, labels = utils.data.read_gt_segmentation(
            data_id, return_labels=True
        )
    else:
        masks, inst_ids, labels = utils.data.read_instpifu_segmentation(
            data_id, return_labels=True
        )

    if len(masks) == 0:
        logger.warning(f"No masks found for {data_id}, skipping")
        return

    instances = Instances((image.shape[-2], image.shape[-1]))
    instances.gt_masks = masks

    batched_inputs = [
        {
            "image": image.to(device),
            "depth": depth.to(device),
            "intrinsics": intrinsics.to(device),
            "instances": instances.to(device),
        }
    ]

    if save_rgb:
        Image.fromarray(pickled_data["rgb_img"]).save(output_path / "rgb.png")
    if save_depth:
        depth_image = ((depth / depth.max()).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(depth_image).save(output_path / "depth.png")

    # Sample triplanes
    for i in range(inf_sample_num):
        embed_path = output_path / f"embed_{i}.npy"
        init_embed_path = output_path / f"init_embed_{i}.npy"
        guided_embed_path = output_path / f"guided_embed_{i}.npz"
        transform_path = output_path / f"guided_transforms_{i}.npz"
        model.scheduler.set_timesteps(50)

        if init_embed_path.exists():
            init_noise = torch.from_numpy(np.load(init_embed_path)).to(device)
        else:
            init_noise = None

        output = model(
            batched_inputs,
            return_step_results=False,
            guidance_scale=None,
            init_noise=init_noise,
        )[0]
        valid_indices = output["gt_indices"]
        pred_embeds = output["pred_embeds"]
        init_noise = output["init_noise"]

        if not init_embed_path.exists():
            np.save(init_embed_path, init_noise.cpu().numpy())

        valid_masks = masks[valid_indices]
        valid_inst_ids = torch.tensor(inst_ids)[valid_indices]

        if not override and embed_path.exists():
            pred_embeds = torch.from_numpy(np.load(embed_path)).to(device)
        else:
            np.save(embed_path, pred_embeds.cpu().numpy())

        if use_guiding:
            if not override and guided_embed_path.exists():
                guided_info = np.load(guided_embed_path)
                guided_embeds = torch.from_numpy(guided_info["guided_embeds"]).to(
                    device
                )
            else:
                valid_guide_indices = torch.arange(len(valid_inst_ids))
                if use_gt_transform:
                    transform_matrices = utils.data.read_gt_transformations(
                        data_id, wrd2cam_matrix, valid_inst_ids
                    )
                    valid_guide_indices = torch.tensor(
                        [i for i, t in enumerate(transform_matrices) if t is not None]
                    )
                    transform_matrices = np.stack(
                        [t for t in transform_matrices if t is not None]
                    )
                elif not transform_path.exists() or override:
                    triplanes = model.denormalize_triplanes(output["triplane"])
                    source_list = utils.geom.triplanes_to_point_clouds(
                        model,
                        triplanes,
                        num_points=5000,
                        sample_method="poisson",
                    )
                    target_list = utils.geom.prepare_depth_point_cloud(
                        depth,
                        valid_masks,
                        intrinsics,
                        num_sample_points=5000,
                        clean_up_with_cluster=False,
                    )
                    transform_matrices, losses = utils.sample.get_scene_transformations(
                        source_list, target_list, intrinsics, return_losses=True
                    )
                    transform_matrices = transform_matrices.numpy()
                    np.savez(
                        transform_path,
                        transforms=transform_matrices,
                        losses=losses.numpy(),
                    )
                else:
                    transform_info = np.load(transform_path)
                    transform_matrices = transform_info["transforms"]

                guided_output = utils.sample.guided_sampling(
                    model,
                    depth.to(device),
                    torch.tensor(transform_matrices).float().to(device),
                    output["triplane_feature"][valid_guide_indices],
                    intrinsics.to(device),
                    valid_masks[valid_guide_indices].to(device),
                    alpha=100,
                    timesteps=100,
                    depth_weight=1,
                    verbose=False,
                    init_noise=init_noise[valid_guide_indices],
                )
                guided_embeds = pred_embeds.clone()
                guided_embeds[valid_guide_indices] = guided_output["pred_embeds"]
                guided_losses = np.zeros(len(pred_embeds))
                guided_losses[valid_guide_indices] = guided_output["last_losses"]
                np.savez(
                    guided_embed_path,
                    guided_embeds=guided_embeds.cpu().numpy(),
                    guided_losses=guided_losses,
                )

        def export_mesh(embeds, prefix):
            triplanes = model.triplane_vae.decode(embeds)
            triplanes = model.denormalize_triplanes(triplanes).to(device)
            scene = utils.geom.triplanes_to_scene(
                model,
                triplanes=triplanes,
                transformations=None,
                resolution=128,
                include_axis=False,
            )
            mesh_out_dir = output_path / f"{prefix}{i}"
            mesh_out_dir.mkdir(parents=True, exist_ok=True)
            for m_idx, mesh in enumerate(scene.geometry.values()):
                mesh.export(mesh_out_dir / f"{m_idx}.ply")

        if save_mesh:
            export_mesh(pred_embeds, "mesh_")
            if use_guiding:
                export_mesh(guided_embeds, "guided_mesh_")


def compute_object_score(
    model,
    output_dir,
    data_id,
    override=False,
    use_guiding=False,
    inf_sample_num=3,
    num_sample_points=10000,  # Default number of sampling points from InstPIFu
):
    output_path = output_dir / f"{data_id}"
    inst_dict_path = output_path / "object_normalized_results.json"

    if inst_dict_path.exists() and not override:
        logger.info(f"Loading results for {data_id}")
        with open(inst_dict_path, "r") as f:
            inst_dict = json.load(f)
        if len(inst_dict) > 0:
            return

    inst_dict = {}

    pred_paths = list(output_path.glob("embed_*.npy"))
    pred_paths = sorted(pred_paths, key=lambda x: int(x.stem.split("_")[-1]))
    if len(pred_paths) > inf_sample_num:
        pred_paths = pred_paths[:inf_sample_num]

    masks, inst_ids, labels = utils.data.read_instpifu_segmentation(
        data_id, return_labels=True
    )
    gt_meshes = utils.data.read_gt_meshes(data_id, inst_ids)

    for sample_id in range(inf_sample_num):
        pred_meshes = _prepare_pred_meshes(
            model,
            output_path,
            sample_id,
            use_guiding=use_guiding,
        )
        # compute transformation
        gt_list = []
        pred_list = []
        for idx in range(len(inst_ids)):
            if gt_meshes[idx] is None:
                logger.warning(
                    f"At {data_id}, instance {inst_ids[idx]} has no ground truth mesh, skipping this instance"
                )
                continue
            if pred_meshes[idx] is None:
                logger.warning(
                    f"At {data_id}, instance {inst_ids[idx]} failed to generate mesh, skipping this instance"
                )
                continue
            gt_pcds = utils.geom.sample_points_from_o3d_mesh(
                gt_meshes[idx], 5000, sample_method="poisson"
            )
            pred_pcds = utils.geom.sample_points_from_o3d_mesh(
                pred_meshes[idx], 5000, sample_method="poisson"
            )
            gt_pcds = utils.geom.get_normalized_pcd(gt_pcds)
            pred_pcds = utils.geom.get_normalized_pcd(pred_pcds)
            gt_list.append(gt_pcds)
            pred_list.append(pred_pcds)

        if len(gt_list) == 0:
            logger.warning(
                f"At {data_id}, no valid ground truth meshes found, skipping this data"
            )
            continue

        transform_matrices = utils.sample.get_object_transformations(pred_list, gt_list)

        valid_count = 0
        for idx in range(len(inst_ids)):
            if gt_meshes[idx] is None or pred_meshes[idx] is None:
                continue
            valid_count += 1
            gt_pcds = utils.geom.sample_points_from_o3d_mesh(
                gt_meshes[idx], num_sample_points
            )
            pred_pcds = utils.geom.sample_points_from_o3d_mesh(
                pred_meshes[idx], num_sample_points
            )
            gt_pcds = utils.geom.get_normalized_pcd(gt_pcds)
            pred_pcds = utils.geom.get_normalized_pcd(pred_pcds)

            transformed_pred_pcds = utils.transforms.apply_transformation_matrix(
                pred_pcds,
                transform_matrices[valid_count - 1],
            )

            cd_loss = chamfer_distance(
                gt_pcds.unsqueeze(0).cuda(),
                transformed_pred_pcds.unsqueeze(0).cuda(),
            )[0].item()

            f_score = utils.metrics.f_score(
                gt_pcds.numpy(), transformed_pred_pcds.numpy()
            )

            if inst_dict.get(inst_ids[idx], None) is None:
                inst_dict[inst_ids[idx]] = {
                    "cd_loss": [],
                    "f_score": [],
                    "label": labels[idx],
                }
            inst_dict[inst_ids[idx]]["cd_loss"].append(cd_loss)
            inst_dict[inst_ids[idx]]["f_score"].append(f_score)

            logger.info(
                f"Data {data_id}, Inst {inst_ids[idx]}: CD Loss {cd_loss:.8f}, F-Score {f_score:.8f}"
            )
    with open(inst_dict_path, "w") as f:
        json.dump(inst_dict, f)


def collate_object_score(output_dir, data_id_iter):
    total_cd_loss = 0.0
    total_f_score = 0.0
    total_count = 0
    result_dict = {}

    for data_id in data_id_iter:
        output_path = output_dir / f"{data_id}"
        inst_dict_path = output_path / "object_normalized_results.json"

        if inst_dict_path.exists():
            with inst_dict_path.open() as fp:
                inst_dict = json.load(fp)

            for inst_id, metric_dict in inst_dict.items():
                label = metric_dict["label"]
                cd_loss = metric_dict["cd_loss"]
                f_score = metric_dict["f_score"]
                total_cd_loss += sum(cd_loss)
                total_f_score += sum(f_score)
                total_count += len(cd_loss)
                if label not in result_dict:
                    result_dict[label] = {"cd_loss": [], "f_score": []}
                result_dict[label]["cd_loss"].extend(cd_loss)
                result_dict[label]["f_score"].extend(f_score)
    return total_cd_loss, total_f_score, total_count, result_dict


def compute_scene_transforms(
    model,
    output_dir,
    data_id,
    depth=None,
    override=False,
    use_guiding=False,
    use_gt_transform=False,
    use_sam=False,
    use_gt_segm=False,
    save_mesh=False,
    inf_sample_num=3,
    cleanup_depth=True,
):
    assert not (
        use_sam and use_gt_transform
    ), "Cannot use both SAM and GT transform at the same time"
    assert not (
        use_sam and use_gt_segm
    ), "Cannot use both SAM and GT segmentation at the same time"

    output_path = output_dir / f"{data_id}"
    pred_paths = list(output_path.glob("embed_*.npy"))
    pred_paths = sorted(pred_paths, key=lambda x: int(x.stem.split("_")[-1]))
    if len(pred_paths) > inf_sample_num:
        pred_paths = pred_paths[:inf_sample_num]

    with open(
        f"datasets/front3d_pifu/data/pickled_data/test/rendertask{data_id}.pkl",
        "rb",
    ) as f:
        pickled_data = pickle.load(f)

    cam_info = pickled_data["camera"]
    scale = cam_info["scale_factor"]
    K = np.asarray(cam_info["K"], dtype=np.float32)
    K[0] /= scale
    K[1] /= scale
    K[2, 2] = 1
    intrinsics = torch.as_tensor(K)
    wrd2cam_matrix = np.asarray(cam_info["wrd2cam_matrix"], dtype=np.float32)

    if use_sam:
        masks, inst_ids, labels = utils.data.read_sam_segmentation(
            data_id, return_labels=True
        )
    elif use_gt_segm:
        masks, inst_ids, labels = utils.data.read_gt_segmentation(
            data_id, return_labels=True
        )
    else:
        masks, inst_ids, labels = utils.data.read_instpifu_segmentation(
            data_id, return_labels=True
        )

    depth = _preprocess_depth(pickled_data, depth)
    target_list = utils.geom.prepare_depth_point_cloud(
        depth,
        masks,
        intrinsics,
        num_sample_points=5000,
        clean_up_with_cluster=cleanup_depth,
    )

    if save_mesh:
        utils.geom.save_colored_pointcloud(
            np.stack(target_list, axis=0), output_path / "scene_depth.ply"
        )

    for sample_id in range(inf_sample_num):
        source_meshes = _prepare_pred_meshes(
            model,
            output_path,
            sample_id,
            use_guiding=use_guiding,
        )
        transforms_path = output_path / f"transforms_{sample_id}.npy"

        if use_gt_transform:
            transform_matrices = utils.data.read_gt_transformations(
                data_id, wrd2cam_matrix, torch.tensor(inst_ids)
            )
        elif transforms_path.exists() and not override:
            transform_matrices = np.load(transforms_path)
        else:
            source_list = []
            valid_target_list = []
            for idx, mesh in enumerate(source_meshes):
                if mesh is None:
                    continue
                source_list.append(
                    utils.geom.sample_points_from_o3d_mesh(mesh, 5000, "poisson")
                )
                valid_target_list.append(target_list[idx])
            transform_matrices = utils.sample.get_scene_transformations(
                source_list, valid_target_list, intrinsics
            )
            np.save(transforms_path, transform_matrices)

        pred_meshes = []

        if len(source_meshes) > len(transform_matrices):
            source_meshes = [m for m in source_meshes if m is not None]

        for idx in range(len(source_meshes)):
            mesh = source_meshes[idx]
            t = transform_matrices[idx]
            if t is None or mesh is None:
                continue
            pred_meshes.append(mesh.transform(t))

        if save_mesh:
            mesh_out_dir = output_path / "mesh_scene"
            mesh_out_dir.mkdir(parents=True, exist_ok=True)
            utils.geom.save_scene_meshes(pred_meshes, mesh_out_dir / f"{sample_id}.glb")


def adaptive_load_state_dict(filepath):
    if filepath.endswith(".safetensors"):
        state_dict = load_safetensors(filepath)
    elif filepath.endswith(".pth") or filepath.endswith(".pt"):
        state_dict = torch.load(filepath, map_location="cpu")
    else:
        raise ValueError("Unsupported file format: Must be .safetensors or .pth")
    return state_dict


def prepare_model(config_path, ckpt_path):
    cfg = utils.inference.setup(
        easydict.EasyDict(
            {
                "config_file": config_path,
                "output": "output/dummy",
                "model": ckpt_path,
            }
        )
    )
    model = build_model(cfg)
    state_dict = adaptive_load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.training = False
    model.cuda()
    return model
