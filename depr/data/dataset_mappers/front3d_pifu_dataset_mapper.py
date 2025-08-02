from typing import List, Union
import os
import copy
import numpy as np
import jsonlines
import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog, detection_utils as utils, transforms as T
from detectron2.structures import BitMasks, Instances
from depr.utils.misc import color_to_id, random_occlude
from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper


__all__ = ["Front3DPIFuDatasetMapper"]


class Front3DPIFuDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        enable_3d: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        enable_depth_noise: bool,
        image_format: str,
        ignore_label,
        num_samples: int,
        size_divisibility,
        dataset_name: str,
        config,
    ):
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility,
        )
        self.enable_3d = enable_3d
        self.dataset_name = dataset_name
        self.metadata = MetadataCatalog.get(dataset_name)
        self.min_instance_pixels = config.MODEL.MODEL_2D.MIN_INSTANCE_PIXELS
        self.num_samples = num_samples
        self.config = config

        if self.config.INPUT.RANDOM_INST_MASK.ENABLED:
            self.inst_mask_prob = self.config.INPUT.RANDOM_INST_MASK.PROB
            # self.inst_masks = glob(os.path.join(self.config.INPUT.RANDOM_INST_MASK.MASK_PATH, "*.png"))
            # self.inst_masks = [utils.read_image(mask_path) / 255 for mask_path in self.inst_masks]
            # self.inst_masks = [mask[:, :, 0] > 0.5 for mask in self.inst_masks]
            # self.inst_masks = torch.stack([torch.from_numpy(mask) for mask in self.inst_masks])
            # torch.save(self.inst_masks, "checkpoint/masks.pt")
            self.inst_masks = torch.load("checkpoint/masks.pt")
        else:
            self.inst_masks = None

        self.enable_depth_noise = enable_depth_noise

        self.stuff_classes = [
            info["id"] for info in self.metadata.class_info if info["isthing"] == 0
        ]

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, enable_3d: bool = False):
        if is_train:
            ret = MaskFormerSemanticDatasetMapper.from_config(cfg, is_train)
            ret["dataset_name"] = list(cfg.DATASETS.TRAIN)[0]
            ret["enable_3d"] = enable_3d
            ret["num_samples"] = cfg.MODEL.MODEL_2D.TRAIN_NUM_SAMPLES
            ret["enable_depth_noise"] = cfg.INPUT.RANDOM_DEPTH_NOISE
        else:
            augs = utils.build_augmentation(cfg, is_train=False)
            dataset_name = cfg.DATASETS.TEST[0]
            meta = MetadataCatalog.get(dataset_name)
            ignore_label = meta.ignore_label
            assert cfg.INPUT.RANDOM_DEPTH_NOISE == False
            assert cfg.INPUT.RANDOM_INST_MASK.ENABLED == False
            ret = {
                "is_train": False,
                "enable_3d": enable_3d,
                "augmentations": augs,
                "dataset_name": dataset_name,
                "image_format": cfg.INPUT.FORMAT,
                "ignore_label": ignore_label,
                "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
                "enable_depth_noise": False,
                "num_samples": cfg.MODEL.MODEL_2D.TRAIN_NUM_SAMPLES,
            }

        ret["config"] = cfg
        return ret

    def __call__(self, dataset_dict: dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image_id = dataset_dict["image_id"]
        pickled_data = np.load(dataset_dict["file_name"], allow_pickle=True)
        image = np.array(pickled_data["rgb_img"])  # 484, 646, 3

        # image = utils.convert_PIL_to_numpy(image, format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            pan_seg_gt = utils.read_image(dataset_dict["segm_label_file_name"])
            pan_seg_gt = color_to_id(pan_seg_gt)
            sem_seg_gt, inst_seg_gt = pan_seg_gt // 1000, pan_seg_gt % 1000
            sem_seg_gt[sem_seg_gt == 0] = self.ignore_label
            sem_seg_gt = sem_seg_gt.astype("double")
            inst_seg_gt = inst_seg_gt.astype("double")

            if self.config.INPUT.PSEUDO_DEPTH.ENABLED:
                depth_gt = (
                    utils.read_image(
                        os.path.join(
                            self.config.INPUT.PSEUDO_DEPTH.DEPTH_PATH, f"{image_id}.png"
                        )
                    )
                    / 65535
                    * 20
                )
            else:
                depth_gt = pickled_data["depth_map"]
                depth_gt = (1 - depth_gt / 255) * 10
            # depth_min = self.config.MODEL.UNI_3D.PROJECTION.DEPTH_MIN
            # depth_max = self.config.MODEL.UNI_3D.PROJECTION.DEPTH_MAX
        else:
            sem_seg_gt = inst_seg_gt = depth_gt = None

        # Apply transformations to the image and annotations
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)

        do_hflip = (
            sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        )

        image, sem_seg_gt = aug_input.image, aug_input.sem_seg  # 484, 646, 3

        if self.is_train:
            inst_seg_gt = transforms.apply_segmentation(inst_seg_gt)

        if depth_gt is not None:
            depth_gt = transforms.apply_segmentation(depth_gt)
            # FIXME: for resize transformation, depth may need to be scaled
            depth_gt = torch.as_tensor(depth_gt.astype("float32"))

        # Pad image and segmentation label here!
        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )  # 3, 484, 646

        if self.is_train:
            inst_seg_gt = torch.as_tensor(inst_seg_gt.astype("long"))
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:  # -1 at here
            raise NotImplementedError(
                "size_divisibility > 0 is not supported, because depth not been tested here"
            )

        height, width = (image.shape[-2], image.shape[-1])

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if self.is_train:
            if self.enable_depth_noise:
                depth_noise = torch.randn_like(depth_gt) * 0.0172  # 3cm ** 0.5
                depth_gt[depth_gt > 0] += depth_noise[depth_gt > 0]

        if self.is_train:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if depth_gt is not None:
            depth_gt = depth_gt.float()
            dataset_dict["depth"] = depth_gt

        if "annotations" in dataset_dict:
            raise ValueError(
                "Panoptic segmentation dataset should not have 'annotations'."
            )

        dataset_dict.pop("instance_label_file_name", None)
        dataset_dict.pop("semantic_label_file_name", None)

        if self.is_train:
            # Prepare per-category binary masks
            indices = torch.unique(inst_seg_gt).tolist()

            instances = Instances((height, width))
            classes = []
            masks = []

            inst_ids = []

            if depth_gt is not None:
                depths = []
                mean_depths = []

            def _add_results(_class_id, _seg_mask):
                # Instance Mask
                if (
                    self.inst_masks is not None
                    and np.random.rand() < self.inst_mask_prob
                ):
                    sampled_mask = self.inst_masks[
                        torch.randint(0, self.inst_masks.shape[0], (1,)).item()
                    ]
                    _seg_mask = random_occlude(
                        _seg_mask,
                        sampled_mask,
                        scale_range=(0.2, 0.9),
                        occlusion_range=(0.05, 0.5),
                    )

                classes.append(_class_id)
                masks.append(_seg_mask)

                if depth_gt is not None:
                    seg_depth = torch.zeros_like(depth_gt)
                    seg_depth[_seg_mask] = depth_gt[_seg_mask]
                    depths.append(seg_depth)
                    valid_seg_depth = seg_depth > 0
                    mean_depths.append(seg_depth.sum() / valid_seg_depth.sum().clamp(1))

            for inst_id in indices:
                if inst_id <= 0:
                    continue

                seg_mask = inst_seg_gt == inst_id
                if seg_mask.sum() < self.min_instance_pixels:
                    continue

                # Determine semantic label of the current instance by voting
                semantic_labels = sem_seg_gt[seg_mask]
                unique_semantic_labels, semantic_label_count = torch.unique(
                    semantic_labels, return_counts=True
                )
                max_semantic_label = torch.argmax(semantic_label_count)
                class_id = unique_semantic_labels[max_semantic_label]

                if class_id == self.ignore_label or class_id in self.stuff_classes:
                    continue

                inst_ids.append(inst_id)
                _add_results(class_id, seg_mask)

            stuff_mask = torch.zeros((height, width), dtype=torch.bool)
            for class_id in self.stuff_classes:
                seg_mask = sem_seg_gt == class_id
                if not seg_mask.any():
                    continue
                stuff_mask[seg_mask] = True
            stuff_available = True
            if stuff_mask.sum() >= self.min_instance_pixels:
                _add_results(12, stuff_mask)  # A large number to represent stuff
            else:
                stuff_available = False

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros(
                    (0, inst_seg_gt.shape[-2], inst_seg_gt.shape[-1])
                )
                instances.gt_classes = torch.zeros(0, dtype=torch.long)
            else:
                masks = BitMasks(torch.stack(masks))
                instances.gt_masks = masks.tensor
                instances.gt_classes = torch.as_tensor(classes)

            if depth_gt is not None:
                if len(masks) == 0:
                    instances.gt_depths = torch.zeros(
                        (0, depth_gt.shape[-2], depth_gt.shape[-1])
                    )
                    instances.mean_depths = torch.zeros(0)
                else:
                    instances.gt_depths = torch.stack(depths)
                    instances.mean_depths = torch.stack(mean_depths)

            dataset_dict["instances"] = instances
            dataset_dict["inst_ids"] = (
                torch.as_tensor(inst_ids) if len(inst_ids) else torch.empty(0)
            )

        if self.enable_3d:
            cam_info = pickled_data["camera"]
            scale = cam_info["scale_factor"]
            wrd2cam_matrix = np.asarray(cam_info["wrd2cam_matrix"], dtype=np.float32)
            K = np.asarray(cam_info["K"], dtype=np.float32)
            K[0] /= scale
            K[1] /= scale
            K[2, 2] = 1

            dataset_dict["world_to_cam"] = torch.as_tensor(wrd2cam_matrix)
            dataset_dict["intrinsics"] = torch.as_tensor(K)
            # dataset_dict["intrinsic"] = torch.eye(4, dtype=torch.float32)
            # dataset_dict["intrinsic"][:3, :3] = torch.as_tensor(K)

            if self.is_train:
                inst_ids_set = set(inst_ids)

                with jsonlines.open(
                    dataset_dict.pop("scene_metadata_file_name")
                ) as reader:
                    instance_metadata = [
                        item for item in reader if item["inst_id"] in inst_ids_set
                    ]
                dataset_dict["instance_metadata"] = instance_metadata

                base_instance_path = dataset_dict.pop("inst_triplane_path")
                base_stuff_path = dataset_dict.pop("stuff_triplane_path")

                inst_id_to_3d = {}
                for instance in instance_metadata:
                    model_id = instance["model_id"]
                    inst_id = instance["inst_id"]
                    instance_path = os.path.join(base_instance_path, f"{model_id}.pkl")
                    if not os.path.exists(instance_path):
                        continue
                    inst_data = np.load(instance_path, allow_pickle=True)
                    inst_id_to_3d[inst_id] = {
                        "emb": inst_data["embedding"],
                        "sdf": inst_data["sdf"],
                        "cube_center": inst_data["cube_center"],
                        "cube_size": inst_data["cube_size"],
                    }

                stuff_3d = None
                stuff_path = os.path.join(base_stuff_path, f"{image_id}.pkl")
                if stuff_available and os.path.exists(stuff_path):
                    stuff_data = np.load(stuff_path, allow_pickle=True)
                    stuff_3d = {
                        "emb": stuff_data["embedding"],
                        "sdf": stuff_data["sdf"],
                        "cube_center": stuff_data["cube_center"],
                        "cube_size": stuff_data["cube_size"],
                    }

                if do_hflip:
                    raise NotImplementedError(
                        "Horizontal flip is not supported for 3D data"
                    )

                # inst_id_to_embedding = {item["inst_id"]: item["embedding"] for item in embeddings}

                gt_embeds = []
                gt_cubes = []
                gt_samples = []
                gt_sdfs = []
                gt_normals = []
                gt_masks = []
                gt_is_3d_valid = []

                def add_empty_data():
                    gt_embeds.append(torch.zeros((2, 32, 96), dtype=torch.float32))
                    gt_cubes.append(
                        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                    )
                    gt_samples.append(
                        torch.zeros(self.num_samples, 3, dtype=torch.float32)
                    )
                    gt_sdfs.append(torch.zeros(self.num_samples, dtype=torch.float32))
                    gt_normals.append(
                        torch.zeros(self.num_samples, 3, dtype=torch.float32)
                    )
                    gt_masks.append(torch.zeros(self.num_samples, dtype=torch.bool))
                    gt_is_3d_valid.append(False)

                def add_data(emb, sdf, cube_center, cube_size):
                    gt_embed = torch.from_numpy(emb)
                    gt_cube = torch.from_numpy(np.append(cube_center, cube_size))
                    gt_sample = torch.from_numpy(sdf["samples"])
                    gt_sdf = torch.from_numpy(sdf["sdf_values"])
                    gt_normal = torch.from_numpy(sdf["normals"])
                    gt_mask = torch.from_numpy(sdf["masks"])

                    gt_embeds.append(gt_embed)
                    gt_cubes.append(gt_cube)
                    gt_samples.append(gt_sample)
                    gt_sdfs.append(gt_sdf)
                    gt_normals.append(gt_normal)
                    gt_masks.append(gt_mask)
                    gt_is_3d_valid.append(True)

                for inst_id in inst_ids:
                    inst_data = inst_id_to_3d.get(inst_id)
                    if inst_data is None:
                        add_empty_data()
                    else:
                        add_data(
                            inst_data["emb"],
                            inst_data["sdf"],
                            inst_data["cube_center"],
                            inst_data["cube_size"],
                        )

                if stuff_available:  # Add stuff SDF
                    if stuff_3d is None:
                        add_empty_data()
                    else:
                        add_data(
                            stuff_3d["emb"],
                            stuff_3d["sdf"],
                            stuff_3d["cube_center"],
                            stuff_3d["cube_size"],
                        )

                instances.gt_embeds = torch.stack(gt_embeds)
                instances.gt_cubes = torch.stack(gt_cubes)
                instances.gt_samples = torch.stack(gt_samples)
                instances.gt_sdfs = torch.stack(gt_sdfs)
                instances.gt_normals = torch.stack(gt_normals)
                instances.gt_masks_3d = torch.stack(gt_masks)
                instances.gt_is_3d_valid = torch.as_tensor(gt_is_3d_valid)

        return dataset_dict
