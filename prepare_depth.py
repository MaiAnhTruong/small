"""VS-Depth depth preprocessing: run Depth-Anything-V2 on a split's images -> 16-bit inverse-depth (disparity)
PNGs that the 3DGS depth pipeline expects. Then run utils/make_depth_scale.py to produce depth_params.json
(robust per-image affine alignment to COLMAP sparse depth). Run on Kaggle (needs a GPU + transformers).

Usage (per scene, TRAIN split only):
  python prepare_depth.py --images_dir <scene>/train/images --out_dir <scene>/train/depths
  python utils/make_depth_scale.py --base_dir <scene>/train --depths_dir <scene>/train/depths --model_type bin
  # then train with:  -d depths   (and --gate_mode uniform)
"""
import os
import argparse
import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="e.g. <scene>/train/depths")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf")
    ap.add_argument("--invert", action="store_true",
                    help="set if the model outputs METRIC depth (we need disparity = inverse depth)")
    args = ap.parse_args()

    import numpy as np
    import cv2
    import torch
    from PIL import Image
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=args.model, device=device)
    os.makedirs(args.out_dir, exist_ok=True)

    imgs = sorted(sum([glob.glob(os.path.join(args.images_dir, e))
                       for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg")], []))
    print(f"[prepare_depth] {len(imgs)} images -> {args.out_dir} (model={args.model})")
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        im = Image.open(p).convert("RGB")
        out = pipe(im)
        d = out["predicted_depth"]
        d = d.squeeze().detach().float().cpu().numpy()           # [H',W'] disparity (larger = closer)
        if args.invert:
            d = 1.0 / np.clip(d, 1e-6, None)
        d = cv2.resize(d, (im.width, im.height), interpolation=cv2.INTER_LINEAR)
        d = d - d.min()
        d = d / max(d.max(), 1e-6)                                # per-image normalize (affine fixed later)
        png16 = (d * 65535.0).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, name + ".png"), png16)
    print("[prepare_depth] done. Next: utils/make_depth_scale.py to build depth_params.json")


if __name__ == "__main__":
    main()
