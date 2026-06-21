"""Evaluate a trained model dir IN-MEMORY (no render-to-disk): load the saved Gaussians + test cameras
(via the model's cfg_args) -> render test views -> print + dump:
  PSNR / SSIM / LPIPS(vgg)            (standard interpolated-holdout metrics; same calls as metrics.py)
  #Gaussians                          (M3 efficiency: quality per capacity)
  geom_consist                        (M2: mean cross-view rendered-depth disagreement; LOWER = cleaner geometry,
                                       fewer floaters -- the geometry signal interp-PSNR is blind to, v8 E1)
Usage:  python eval_metrics.py -m <model_dir>
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.covisibility import _CovCam, unproject_depth, project_points, _sample
from arguments import ModelParams, PipelineParams, get_combined_args


@torch.no_grad()
def geom_consistency(cams, depths_z, max_dim=128):
    """M2: mean relative cross-view depth disagreement of the rendered geometry (no GT). For each view, unproject
    its (downsampled) depth, reproject into every other view, compare to that view's depth. LOWER = the surface
    is more multi-view-consistent (fewer floaters/haze). Reuses the verified projection primitives."""
    low_c, low_d = [], []
    for c, dz in zip(cams, depths_z):
        H, W = dz.shape
        s = max(1, int(round(max(H, W) / float(max_dim))))
        Hl, Wl = max(2, H // s), max(2, W // s)
        low_c.append(_CovCam(c, Wl, Hl))
        low_d.append(F.interpolate(dz[None, None], size=(Hl, Wl), mode="nearest")[0, 0])
    V = len(low_c)
    res = []
    for i in range(V):
        Xi = unproject_depth(low_c[i], low_d[i]).reshape(-1, 3)
        zi = low_d[i].reshape(-1).clamp_min(1e-6)
        acc = torch.zeros_like(zi); cnt = torch.zeros_like(zi)
        for j in range(V):
            if j == i:
                continue
            u, v, zX = project_points(Xi, low_c[j])
            Wj, Hj = int(low_c[j].image_width), int(low_c[j].image_height)
            infr = (u >= 0) & (u < Wj) & (v >= 0) & (v < Hj) & (zX > 1e-6)
            zj = _sample(low_d[j], u.clamp(0, Wj - 1), v.clamp(0, Hj - 1), Wj, Hj)
            r = (zX - zj).abs() / zi
            acc += torch.where(infr, r, torch.zeros_like(r)); cnt += infr.float()
        m = cnt > 0
        if m.any():
            res.append((acc[m] / cnt[m]))
    return float(torch.cat(res).mean()) if res else float("nan")


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

    psnrs, ssims, lpipss, depths = [], [], [], []
    for v in test:
        pkg = render(v, g, pipe, bg, use_trained_exp=ds.train_test_exp)
        im = torch.clamp(pkg["render"], 0.0, 1.0)
        gt = torch.clamp(v.original_image.cuda(), 0.0, 1.0)
        dz = (1.0 / pkg["depth"][0].clamp_min(1e-6)).detach()
        H, W = dz.shape; s = max(1, int(round(max(H, W) / 256.0)))             # store depth small (M2 bounded mem)
        depths.append(F.interpolate(dz[None, None], size=(max(2, H // s), max(2, W // s)), mode="nearest")[0, 0])
        if ds.train_test_exp:                              # eval right half only (matches training_report)
            im = im[..., im.shape[-1] // 2:]; gt = gt[..., gt.shape[-1] // 2:]
        psnrs.append(psnr(im[None], gt[None]).item())
        ssims.append(ssim(im[None], gt[None]).item())
        lpipss.append(lpips(im[None], gt[None], net_type='vgg').item())

    geom = geom_consistency(test, depths)                                      # M2
    res = {"model": os.path.basename(ds.model_path.rstrip("/")), "iter": scene.loaded_iter,
           "n_gauss": int(g.get_xyz.shape[0]), "n_test": len(test),
           "PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims)), "LPIPS_vgg": float(np.mean(lpipss)),
           "geom_consist": geom}
    with open(os.path.join(ds.model_path, "metrics_vgg.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[METRICS] {res['model']}  iter={res['iter']}  #Gauss={res['n_gauss']}  "
          f"PSNR={res['PSNR']:.3f}  SSIM={res['SSIM']:.4f}  LPIPS_vgg={res['LPIPS_vgg']:.4f}  "
          f"geom_consist={res['geom_consist']:.4f}  (test n={res['n_test']})", flush=True)


if __name__ == "__main__":
    main()
