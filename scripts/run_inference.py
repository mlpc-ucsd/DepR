import jsonlines
from pathlib import Path
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
from depr import utils
import argparse


torch.autograd.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True, choices=["sample", "score", "collate", "scene"]
    )
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--config", type=str, default="checkpoint/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/unet.safetensors")
    parser.add_argument("--metadata", type=str, default="")
    parser.add_argument("--guided", action="store_true", help="Use guided sampling")
    parser.add_argument("--use-gt-layout", action="store_true", help="Use ground truth layout")
    parser.add_argument("--use-gt-depth", action="store_true", help="Use ground truth depth")
    parser.add_argument("--use-gt-segm", action="store_true", help="Use ground truth segmentation")
    parser.add_argument("--output-mesh", action="store_true", help="Export mesh files via Marching Cubes")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples per object")
    parser.add_argument("--override", action="store_true", help="Override existing scores")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM for segmentation (needed by scene-level evaluation)")
    parser.add_argument("--cleanup-depth", action="store_true", help="Clean up depth point clouds (Scene mode only)")
    parser.add_argument("--output-dir", type=str, default="output/infer", help="Output directory for results")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)

    args = parser.parse_args()

    exp_name = ""
    if args.guided:
        exp_name = "guided_"
    if args.use_gt_segm:
        exp_name += "gt_segm_"
    if args.use_gt_layout:
        exp_name += "gt_layout_"
    if args.use_sam:
        exp_name += "sam_"
    if args.use_gt_depth:
        exp_name += "gt_depth_"
    if args.exp_name != "":
        exp_name += args.exp_name + "_"
    exp_name += "depr"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        output=str(output_dir / f"{args.mode}.log"),
        color=False,
    )

    if args.mode != "collate":
        model = utils.inference.prepare_model(args.config, args.checkpoint)

    metadata = args.metadata
    if not metadata:
        if args.mode == "scene":
            metadata = "datasets/front3d_pifu/meta/test_scene.jsonl"
        else:
            metadata = "datasets/front3d_pifu/meta/test_obj.jsonl"

    with jsonlines.open(metadata) as reader:
        data_list = [item["image_id"] for item in reader]

    data_list = data_list[args.rank :: args.world_size]

    pbar = tqdm(data_list)

    if args.mode == "sample":
        for data_id in pbar:
            try:
                if args.use_gt_depth:
                    depth = None
                else:
                    depth = utils.data.read_depth_pro_depth(data_id)
                utils.inference.reconstruct_object(
                    model=model,
                    output_dir=output_dir,
                    data_id=data_id,
                    depth=depth,
                    use_guiding=args.guided,
                    use_gt_transform=args.use_gt_layout,
                    use_sam=args.use_sam,
                    use_gt_segm=args.use_gt_segm,
                    inf_sample_num=args.num_samples,
                    override=args.override,
                    save_rgb=True,
                    save_depth=True,
                    save_mesh=args.output_mesh,
                )
            except Exception as e:
                logger.warning(f"Error processing {data_id}: {e}", exc_info=True)
                raise e
    elif args.mode == "score":
        if args.use_sam:
            raise ValueError("`--use-sam` is not supported in `score` mode")
        if args.use_gt_segm:
            raise ValueError("`--use-gt-segm` is not supported in `score` mode")
        for data_id in pbar:
            try:
                utils.inference.compute_object_score(
                    model=model,
                    output_dir=output_dir,
                    data_id=data_id,
                    inf_sample_num=args.num_samples,
                    use_guiding=args.guided,
                    override=args.override,
                )
            except Exception as e:
                logger.warning(f"Error processing {data_id}: {e}", exc_info=True)
                raise e
    elif args.mode == "collate":
        if args.use_sam:
            raise ValueError("`--use-sam` is not supported in `collate` mode")
        if args.use_gt_segm:
            raise ValueError("`--use-gt-segm` is not supported in `collate` mode")
        total_cd_loss, total_f_score, total_count, result_dict = (
            utils.inference.collate_object_score(
                output_dir=output_dir,
                data_id_iter=pbar,
            )
        )
        pifu_meta = MetadataCatalog.get("front3d_pifu_test")

        logger.info(f"Results for {exp_name}")
        logger.info(f"Total Count: {total_count}")
        logger.info(f"Average CD Loss: {total_cd_loss / max(1, total_count):.8f}")
        logger.info(f"Average F-Score: {total_f_score / max(1, total_count):.8f}")

        result_items = sorted(
            result_dict.items(), key=lambda x: x[0]
        )  # sort by class ID
        for key, value in result_items:
            name = pifu_meta.class_info[key - 1]["name"]
            logger.info(f"Class: {name}")
            num_samples = len(value["cd_loss"])
            logger.info(f"  Count: {num_samples}")
            logger.info(
                f"  Average CD Loss: {sum(value['cd_loss']) / max(1, num_samples):.8f}"
            )
            logger.info(
                f"  Average F-Score: {sum(value['f_score']) / max(1, len(value['f_score'])):.8f}"
            )
    elif args.mode == "scene":
        for data_id in pbar:
            if args.use_gt_layout and args.use_sam:
                raise ValueError("`--use-gt-layout` and `--use-sam` cannot be used together in `scene` mode")
            if args.use_gt_segm and args.use_sam:
                raise ValueError("`--use-gt-segm` and `--use-sam` cannot be used together in `scene` mode")
            try:
                if args.use_gt_depth:
                    depth = None
                else:
                    depth = utils.data.read_depth_pro_depth(data_id)
                utils.inference.compute_scene_transforms(
                    model=model,
                    output_dir=output_dir,
                    data_id=data_id,
                    depth=depth,
                    override=args.override,
                    use_guiding=args.guided,
                    use_gt_transform=args.use_gt_layout,
                    use_sam=args.use_sam,
                    use_gt_segm=args.use_gt_segm,
                    save_mesh=args.output_mesh,
                    inf_sample_num=args.num_samples,
                    cleanup_depth=args.cleanup_depth,
                )
            except Exception as e:
                logger.warning(f"Error processing {data_id}: {e}", exc_info=True)
                raise e


if __name__ == "__main__":
    main()
