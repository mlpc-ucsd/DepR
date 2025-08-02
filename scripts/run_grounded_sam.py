import argparse
import cv2
import json
import jsonlines
import numpy as np
import supervision as sv
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

torch.autograd.set_grad_enabled(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "desk",
    "dresser",
    "lamp",
    "nightstand",
    "bookshelf",
]


def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gdino-config",
        type=str,
        default="checkpoint/grounded_sam/GroundingDINO_SwinB.py",
    )
    parser.add_argument(
        "--gdino-weight",
        type=str,
        default="checkpoint/grounded_sam/groundingdino_swinb_cogcoor.pth",
    )
    parser.add_argument("--sam-encoder", type=str, default="vit_h")
    parser.add_argument(
        "--sam-weight", type=str, default="checkpoint/grounded_sam/sam_vit_h_4b8939.pth"
    )
    parser.add_argument("--box-thres", type=float, default=0.3)
    parser.add_argument("--text-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.8)
    parser.add_argument(
        "--metadata", type=str, default="datasets/front3d_pifu/meta/test_obj.jsonl"
    )
    parser.add_argument(
        "--img-path", type=str, default="datasets/front3d_pifu/data/img"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/front3d_pifu/data/grounded_sam"
    )
    args = parser.parse_args()

    # Building GroundingDINO inference model
    grounding_dino_model = Model(
        model_config_path=args.gdino_config, model_checkpoint_path=args.gdino_weight
    )

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[args.sam_encoder](checkpoint=args.sam_weight)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    with jsonlines.open(args.metadata) as reader:
        data_list = [item["image_id"] for item in reader]

    img_folder = Path(args.img_path)

    for uid in tqdm(data_list):
        image_path = img_folder / f"{uid}.png"
        out_path = Path(args.output) / f"{uid}"
        out_path.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(image_path.as_posix())

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=args.box_thres,
            text_threshold=args.text_thres,
        )

        # annotate image with detections
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]
        annotated_frame = box_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # save the annotated grounding dino image
        cv2.imwrite((out_path / "annotated_det.jpg").as_posix(), annotated_frame)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                args.nms_thres,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        nms_idx = (
            torchvision.ops.batched_nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                torch.from_numpy(detections.class_id),
                0.6,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After category-specific NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]
        annotated_image = mask_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
        # save the annotated grounded-sam image
        cv2.imwrite((out_path / "annotated_sam.jpg").as_posix(), annotated_image)

        np.savez_compressed(out_path / "masks.npz", masks=detections.mask)
        label_dict = []
        for i in range(len(detections)):
            class_id = int(detections.class_id[i])
            label_dict.append(
                {
                    "class_id": class_id,
                    "class_name": CLASSES[class_id],
                    "confidence": float(detections.confidence[i]),
                    "box": detections.xyxy[i].tolist(),
                }
            )
        with (out_path / "mask_annotations.json").open("w") as f:
            json.dump(label_dict, f)


if __name__ == "__main__":
    main()
