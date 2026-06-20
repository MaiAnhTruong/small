"""Evaluate a trained model dir IN-MEMORY (no render-to-disk): load the saved Gaussians + test cameras
(via the model's cfg_args) -> render test views -> print + dump PSNR / SSIM / LPIPS(vgg) + #Gaussians.
Same metric calls as metrics.py (4D [1,3,H,W]). Usage:  python eval_metrics.py -m <model_dir>
"""
import os, json
import numpy as np
import torch
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from arguments import ModelParams, PipelineParams, get_combined_args


@torch.no_grad()
def main():
    parser = ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    args = get_combined_args(parser)                       # reads <model>/cfg_args (source_path, depths, eval, ...)
    ds = lp.extract(args); pipe = pp.extract(args)

    g = GaussianModel(ds.sh_degree, "default")
    scene = Scene(ds, g, load_iteration=-1, shuffle=False) # load latest saved iteration
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    test = scene.getTestCameras()

    psnrs, ssims, lpipss = [], [], []
    for v in test:
        im = torch.clamp(render(v, g, pipe, bg, use_trained_exp=ds.train_test_exp)["render"], 0.0, 1.0)
        gt = torch.clamp(v.original_image.cuda(), 0.0, 1.0)
        if ds.train_test_exp:                              # eval right half only (matches training_report)
            im = im[..., im.shape[-1] // 2:]; gt = gt[..., gt.shape[-1] // 2:]
        psnrs.append(psnr(im[None], gt[None]).item())
        ssims.append(ssim(im[None], gt[None]).item())
        lpipss.append(lpips(im[None], gt[None], net_type='vgg').item())

    res = {"model": os.path.basename(ds.model_path.rstrip("/")), "iter": scene.loaded_iter,
           "n_gauss": int(g.get_xyz.shape[0]), "n_test": len(test),
           "PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims)), "LPIPS_vgg": float(np.mean(lpipss))}
    with open(os.path.join(ds.model_path, "metrics_vgg.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[METRICS] {res['model']}  iter={res['iter']}  #Gauss={res['n_gauss']}  "
          f"PSNR={res['PSNR']:.3f}  SSIM={res['SSIM']:.4f}  LPIPS_vgg={res['LPIPS_vgg']:.4f}  "
          f"(test n={res['n_test']})", flush=True)


if __name__ == "__main__":
    main()
