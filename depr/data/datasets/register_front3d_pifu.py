import jsonlines
import logging
from pathlib import Path
import os
from detectron2.data import DatasetCatalog, MetadataCatalog


FRONT3D_PIFU_CATEGORIES = [
    {"color": (220, 20, 60), "isthing": 1, "id": 1, "trainId": 1, "name": "cabinet"},
    {"color": (255, 0, 0), "isthing": 1, "id": 2, "trainId": 2, "name": "bed"},
    {"color": (0, 0, 142), "isthing": 1, "id": 3, "trainId": 3, "name": "chair"},
    {"color": (0, 0, 70), "isthing": 1, "id": 4, "trainId": 4, "name": "sofa"},
    {"color": (0, 60, 100), "isthing": 1, "id": 5, "trainId": 5, "name": "table"},
    {"color": (0, 80, 100), "isthing": 1, "id": 6, "trainId": 6, "name": "desk"},
    {"color": (0, 0, 230), "isthing": 1, "id": 7, "trainId": 7, "name": "dresser"},
    {
        "color": (119, 11, 32),
        "isthing": 1,
        "id": 8,
        "trainId": 8,
        "name": "lamp",
    },  # Unused in InstPIFu
    {
        "color": (93, 165, 236),
        "isthing": 1,
        "id": 9,
        "trainId": 9,
        "name": "nightstand",
    },  # Added from InstPIFu
    {
        "color": (228, 118, 184),
        "isthing": 1,
        "id": 10,
        "trainId": 10,
        "name": "bookshelf",
    },  # Added from InstPIFu
    {"color": (190, 50, 60), "isthing": 1, "id": 11, "trainId": 11, "name": "other"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 12, "name": "wall"},
    {"color": (128, 64, 128), "isthing": 0, "id": 13, "trainId": 13, "name": "floor"},
    {"color": (70, 70, 70), "isthing": 0, "id": 14, "trainId": 14, "name": "ceiling"},
]


logger = logging.getLogger(__name__)


_ALL_SPLITS = ["train", "val", "test"]
_FRONT_3D_PIFU_BASE_DIR = "front3d_pifu"


def load_front3d_pifu(data_dir: str, metadata_file: str, enable_3d: bool, split: str):
    data_dir = Path(data_dir)
    assert data_dir.exists(), data_dir + " not exists"

    ret = []

    with jsonlines.open(metadata_file) as reader:
        for file_dict in reader:
            image_id = file_dict["image_id"]
            item = {
                "inst_triplane_path": os.path.join(data_dir, "instance_triplane"),
                "stuff_triplane_path": os.path.join(
                    data_dir, "stuff_triplane", f"{image_id}.pkl"
                ),
                "file_name": os.path.join(
                    data_dir, "pickled_data", split, f"rendertask{image_id}.pkl"
                ),
                "segm_label_file_name": os.path.join(
                    data_dir, "panoptic", f"{image_id}.png"
                ),
            }
            item = {
                **item,
                "image_id": image_id,
                "scene_metadata_file_name": os.path.join(
                    data_dir, "metadata", f"{image_id}.jsonl"
                ),
                "height": file_dict["height"],
                "width": file_dict["width"],
            }
            ret.append(item)

    assert len(ret), f"No samples found in {data_dir}!"
    logger.info("Loaded {} samples from {}".format(len(ret), data_dir))
    return ret


def register_all_front_3d_pifu(root):
    meta = {}

    thing_classes = [k["name"] for k in FRONT3D_PIFU_CATEGORIES]
    thing_colors = [k["color"] for k in FRONT3D_PIFU_CATEGORIES]
    stuff_classes = [k["name"] for k in FRONT3D_PIFU_CATEGORIES]
    stuff_colors = [k["color"] for k in FRONT3D_PIFU_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in FRONT3D_PIFU_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for split in _ALL_SPLITS:
        image_dir = os.path.join(root, _FRONT_3D_PIFU_BASE_DIR, "data")

        for flavor in ("_2d", ""):
            data_type = split + flavor
            key = f"front3d_pifu_{data_type}"
            metadata_json = os.path.join(
                root, _FRONT_3D_PIFU_BASE_DIR, "meta", data_type + ".jsonl"
            )
            enable_3d = "2d" not in flavor
            DatasetCatalog.register(
                key,
                lambda x=image_dir, y=metadata_json, z=enable_3d, a=split: load_front3d_pifu(
                    x, y, z, a
                ),
            )
            MetadataCatalog.get(key).set(
                image_root=image_dir,
                gt_dir=metadata_json,
                evaluator_type=(
                    "front3d_pifu_panoptic" if not enable_3d else "front3d_pifu"
                ),
                ignore_label=255,
                label_divisor=1000,
                class_info=FRONT3D_PIFU_CATEGORIES,
                **meta,
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_front_3d_pifu(_root)
