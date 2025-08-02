import tqdm
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import depth_pro


def initialize_depth_pro():
    config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = "checkpoint/depth_pro.pt"
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    model.cuda()

    def get_depth_from_depth_pro(image: np.ndarray, f_px: float) -> torch.Tensor:
        image = transform(image).cuda()
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        return depth.detach().cpu()

    return get_depth_from_depth_pro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="datasets/front3d_pifu/data/depth/depth_pro")
    args = parser.parse_args()

    get_depth_from_depth_pro = initialize_depth_pro()
    data_dirs = sorted(
        Path("datasets/front3d_pifu/data/pickled_data").glob(f"*/rendertask*.pkl"),
        key=lambda x: int(x.stem[10:]),
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_dir in tqdm.tqdm(data_dirs):
        with data_dir.open("rb") as f:
            pickled_data = pickle.load(f)
        data_id = data_dir.stem[10:]
        cam_info = pickled_data["camera"]
        scale = cam_info["scale_factor"]
        K = np.asarray(cam_info["K"], dtype=np.float32)
        K[0] /= scale
        K[1] /= scale
        K[2, 2] = 1
        depth = get_depth_from_depth_pro(pickled_data["rgb_img"], K[0, 0])
        # Save depth as 16-bit PNG, mapping values from [0, 20] to [0, 65535]
        depth = (depth / 20 * 65535).numpy().astype(np.uint16)
        depth = Image.fromarray(depth)
        depth.save(output_dir / f"{data_id}.png")

if __name__ == "__main__":
    main()
