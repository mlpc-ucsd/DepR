from typing import List
import os
import numpy as np
import glob
import pickle
import cv2
import scipy
from scipy.spatial.transform import Rotation
import open3d as o3d
import json
import jsonlines
import torch
from detectron2.data import MetadataCatalog, detection_utils
from depr.utils.misc import color_to_id


pifu_meta = MetadataCatalog.get("front3d_pifu_test")
pifu_name_to_id = {item["name"]: item["id"] for item in pifu_meta.class_info}


DATASET_PREFIX = "datasets/front3d_pifu/data"


def read_binary_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 128
    return mask


def read_gt_segmentation(
    data_id: int,
    return_labels=False,
    include_stuff=False,
    exclude_lamp=False,
    area_threshold=1000,
):
    pan_seg_gt = detection_utils.read_image(f"{DATASET_PREFIX}/panoptic/{data_id}.png")
    pan_seg_gt = color_to_id(pan_seg_gt)
    sem_seg, inst_seg = pan_seg_gt // 1000, pan_seg_gt % 1000

    sem_seg = torch.from_numpy(sem_seg.astype(np.int64)).long()
    inst_seg = torch.from_numpy(inst_seg.astype(np.int64)).long()

    indices = torch.unique(inst_seg).tolist()
    stuff_mask = torch.zeros_like(inst_seg, dtype=torch.bool)
    masks = []
    inst_ids = []
    labels = []
    for i in indices:
        if i <= 0:
            continue
        mask = inst_seg == i
        if mask.sum() < area_threshold:
            continue

        semantic_labels = sem_seg[mask]
        unique_semantic_labels, semantic_label_count = torch.unique(
            semantic_labels, return_counts=True
        )
        max_semantic_label = torch.argmax(semantic_label_count)
        class_id = unique_semantic_labels[max_semantic_label]
        if class_id >= 12:
            stuff_mask |= mask
            continue

        mask_indices = torch.nonzero(mask, as_tuple=False)
        x_1, y_1 = mask_indices.min(dim=0)[0]
        x_2, y_2 = mask_indices.max(dim=0)[0]
        # bbox should larger than 10x10
        if x_2 - x_1 < 10 or y_2 - y_1 < 10:
            continue

        inst_ids.append(int(i))
        labels.append(int(class_id))
        masks.append(mask.bool())

    if include_stuff:
        masks.append(stuff_mask)
        inst_ids.append(-1)
        labels.append(12)

    if exclude_lamp:
        valid_indices = [idx for idx, label in enumerate(labels) if label != 8]
        masks = [masks[idx] for idx in valid_indices]
        inst_ids = [inst_ids[idx] for idx in valid_indices]
        labels = [labels[idx] for idx in valid_indices]

    masks = torch.stack(masks, dim=0)
    if return_labels:
        return masks, inst_ids, labels
    return masks, inst_ids


def read_instpifu_segmentation(data_id: int, include_stuff=False, return_labels=False):
    pifu_paths = glob.glob(f"{DATASET_PREFIX}/instpifu_mask/rendertask{data_id}_*.png")

    masks, inst_ids, labels = read_gt_segmentation(
        data_id,
        return_labels=True,
        include_stuff=False,
        area_threshold=1600,
    )

    pifu_masks = []
    result_labels = []
    result_masks = []
    result_inst_ids = []

    for idx, pifu_path in enumerate(pifu_paths):
        pifu_mask = torch.from_numpy(read_binary_mask(pifu_path))
        pifu_masks.append(pifu_mask)

    # Use linear_sum_assignment to match the masks
    cost_matrix = torch.zeros((len(pifu_masks), len(masks)))
    for i, pifu_mask in enumerate(pifu_masks):
        for j, mask in enumerate(masks):
            # calculate the intersection over union
            cost_matrix[i, j] = (
                (pifu_mask & mask).sum() / (pifu_mask | mask).sum()
            ).float()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-cost_matrix)

    for i, j in zip(row_ind, col_ind):
        result_masks.append(masks[j])
        result_inst_ids.append(inst_ids[j])
        result_labels.append(labels[j])

    if include_stuff:
        raise NotImplementedError

    if return_labels:
        return torch.stack(result_masks, dim=0), result_inst_ids, result_labels
    return torch.stack(result_masks, dim=0), result_inst_ids


def read_sam_segmentation(data_id, return_labels=False, area_threshold=1000, sam_path=None):
    if sam_path is None:
        sam_path = f"{DATASET_PREFIX}/grounded_sam/{data_id}"
    else:
        sam_path = os.path.join(sam_path, str(data_id))
    with open(os.path.join(sam_path, "mask_annotations.json"), "r") as f:
        mask_annotations = json.load(f)

    valid_mask_annotations = []

    all_masks = np.load(os.path.join(sam_path, "masks.npz"))["masks"]

    for idx, item in enumerate(mask_annotations):
        item["mask"] = all_masks[idx]
        item["inst_id"] = idx
        # remove small mask
        if item["mask"].sum() < area_threshold:
            continue
        valid_mask_annotations.append(item)

    inst_ids = []
    masks = []
    labels = []
    for item in valid_mask_annotations:
        inst_ids.append(item["inst_id"])
        masks.append(torch.from_numpy(item["mask"]))
        labels.append(pifu_name_to_id[item["class_name"]])
    if len(masks) > 0:
        masks = torch.stack(masks, dim=0)
    if return_labels:
        return masks, inst_ids, labels
    return masks, inst_ids


def read_depth_pro_depth(data_id):
    dp_depth = (
        detection_utils.read_image(
            os.path.join(f"{DATASET_PREFIX}/depth/depth_pro/", f"{data_id}.png")
        )
        / 65535
        * 20
    )
    dp_depth = torch.from_numpy(dp_depth).float()
    return dp_depth


def read_decube_transformations(data_id, inst_ids):
    sdf_file_name = f"{DATASET_PREFIX}/sdf_fullres_obj/{data_id}.npz"
    sdf_data = np.load(sdf_file_name, allow_pickle=True)["data"]

    def get_decube(sdf):
        decube_matrix = np.eye(4, dtype=np.float32)
        decube_matrix[:3, :3] *= sdf["cube_size"] / 2
        decube_matrix[:3, 3] = sdf["cube_center"]
        return decube_matrix

    obj2cam_dict = {}

    for sdf in sdf_data[:-1]:
        decube_matrix = get_decube(sdf)
        obj2cam_dict[sdf["inst_id"]] = decube_matrix

    obj2cam_list = [
        obj2cam_dict.get(int(inst_id)) for inst_id in inst_ids if int(inst_id) != -1
    ] + ([get_decube(sdf_data[-1])] if -1 == int(inst_ids[-1]) else [])

    return obj2cam_list


def read_gt_meshes(data_id: int, inst_ids) -> list[o3d.geometry.TriangleMesh]:
    metadata_file_name = f"{DATASET_PREFIX}/metadata/{data_id}.jsonl"
    inst_ids = [int(inst_id) for inst_id in inst_ids]
    with jsonlines.open(metadata_file_name) as reader:
        instance_metadata = [item for item in reader if item["inst_id"] in inst_ids]
    inst_to_3d = {}
    for inst in instance_metadata:
        model_id = inst["model_id"]
        model_path = (
            f"{DATASET_PREFIX}/3D-FUTURE-watertight/{model_id}/raw_watertight.obj"
        )
        if inst_to_3d.get(inst["inst_id"]) is None and os.path.exists(model_path):
            inst_to_3d[inst["inst_id"]] = o3d.io.read_triangle_mesh(model_path)
    mesh_list = [inst_to_3d.get(inst_id, None) for inst_id in inst_ids]
    return mesh_list


def read_gt_transformations(data_id, w2cam, inst_ids, apply_decube=True):
    sdf_file_name = f"{DATASET_PREFIX}/sdf_layout/{data_id}.npy"
    sdf_data = np.load(sdf_file_name, allow_pickle=True)

    def get_decube(sdf):
        decube_matrix = np.eye(4, dtype=np.float32)
        decube_matrix[:3, :3] *= sdf["cube_size"] / 2
        decube_matrix[:3, 3] = sdf["cube_center"]
        return decube_matrix

    obj2cam_dict = {}
    y_up_matrix = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))

    for sdf in sdf_data[:-1]:
        obj2wrd_matrix = np.eye(4, dtype=np.float32)
        R = Rotation.from_quat(sdf["rot"]).as_matrix()
        obj2wrd_matrix[:3, :3] = R
        obj2wrd_matrix[:3, 3] = sdf["pos"]
        scale_matrix = np.diag(sdf["scale"] + [1]).astype(np.float32)
        decube_matrix = get_decube(sdf)
        obj2cam_dict[sdf["inst_id"]] = (
            y_up_matrix @ w2cam @ obj2wrd_matrix @ scale_matrix
        )
        if apply_decube:
            obj2cam_dict[sdf["inst_id"]] = obj2cam_dict[sdf["inst_id"]] @ decube_matrix

    obj2cam_list = [
        obj2cam_dict.get(int(inst_id)) for inst_id in inst_ids if int(inst_id) != -1
    ] + ([get_decube(sdf_data[-1])] if -1 == int(inst_ids[-1]) else [])

    return obj2cam_list


def build_gt_scene(data_id, exclude_lamp=False) -> List[o3d.geometry.TriangleMesh]:
    with open(
        f"{DATASET_PREFIX}/pickled_data/test/rendertask{data_id}.pkl",
        "rb",
    ) as f:
        pickled_data = pickle.load(f)

    masks, inst_ids, labels = read_gt_segmentation(
        data_id, return_labels=True, exclude_lamp=exclude_lamp
    )
    gt_meshes = read_gt_meshes(data_id, inst_ids)
    cam_info = pickled_data["camera"]
    wrd2cam_matrix = np.asarray(cam_info["wrd2cam_matrix"], dtype=np.float32)
    gt_transform_matrices = read_gt_transformations(
        data_id, wrd2cam_matrix, torch.tensor(inst_ids), apply_decube=False
    )

    output = []
    inst_count = 0
    for idx in range(len(inst_ids)):
        if gt_meshes[idx] is None or gt_transform_matrices[idx] is None:
            continue
        inst_count += 1
        transformed = gt_meshes[idx].transform(gt_transform_matrices[idx])
        output.append(transformed)
    return output
