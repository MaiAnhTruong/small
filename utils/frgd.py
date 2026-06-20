#
# VS-Depth v4 — FRGD: Fisher-Reliability-Guided Densification (core geometry, CPU-testable).
#
# Rationale (DESIGN_AND_PROOF_v4.md, CPU-verified T3/T4/T6):
#   - depth-LOSS is zero-sum + needs δ<δ_thresh (FAILS in low-H/high-δ) -> measured dead.
#   - 3DGS densifies ∝ texture (√H) -> UNDER-represents low-texture regions (T6: 33% area, 1% points).
#   - multi-view fusion reduces mono-depth error (T4: -20% at SfM truth) -> D_ref usable to PLACE points.
#   FRGD ADDS Gaussians (non-zero-sum) at refined depth D_ref in under-reconstructed + reliable regions.
#
# This module = the SAFE CPU-testable core: (1) refine_depth_maps (cross-view robust fusion -> D_ref + rel),
# (2) generate_frgd_points (select candidate pixels, back-project to 3D, sample color). The train-time hook
# (detect holes via render, add Gaussians via densification_postfix) lives in train.py / gaussian_model.
#
import torch
import torch.nn.functional as F
from utils.covisibility import project_points, unproject_depth, _sample, _intrinsics, _CovCam


def unproject_pixels(cam, u, v, z):
    """pixels (u,v)[N] + camera-z z[N] -> world xyz [N,3]. Inverse of project_points."""
    fx, fy, cx, cy, W, H = _intrinsics(cam)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    cam_h = torch.stack([x, y, z, torch.ones_like(z)], dim=1)                  # [N,4]
    world_h = cam_h @ torch.inverse(cam.world_view_transform)
    return world_h[:, :3]


@torch.no_grad()
def refine_depth_maps(cams, mono_depths_z, tau=0.05, max_dim=200):
    """Cross-view robust fusion of ALIGNED mono depth -> refined depth + reliability, per view.
    mono_depths_z[i]: [H,W] METRIC depth from aligned mono (= 1/aligned_invdepth). Defined everywhere
    (incl. textureless holes), so usable to place points where there are no SfM/render surfaces.
    Returns D_ref[i] [H,W] (refined metric depth) and rel[i] [H,W] in (0,1] (high = views agree)."""
    V = len(cams)
    low, lowd = [], []
    for i in range(V):
        Hh, Ww = mono_depths_z[i].shape
        s = max(1, int(round(max(Hh, Ww) / float(max_dim))))
        Hl, Wl = max(2, Hh // s), max(2, Ww // s)
        low.append(_CovCam(cams[i], Wl, Hl))
        lowd.append(F.interpolate(mono_depths_z[i][None, None], size=(Hl, Wl), mode="nearest")[0, 0])
    D_ref, REL = [], []
    for i in range(V):
        Hl, Wl = lowd[i].shape
        Xi = unproject_depth(low[i], lowd[i]).reshape(-1, 3)                   # candidate 3D from own mono
        acc = [Xi]                                                            # list of 3D estimates per view
        for j in range(V):
            if j == i:
                continue
            u, v, z = project_points(Xi, low[j])
            Wj, Hj = int(low[j].image_width), int(low[j].image_height)
            inframe = (u >= 0) & (u < Wj) & (v >= 0) & (v < Hj) & (z > 1e-6)
            dj = _sample(lowd[j], u.clamp(0, Wj - 1), v.clamp(0, Hj - 1), Wj, Hj)   # neighbor mono depth
            Xj = unproject_pixels(low[j], u.clamp(0, Wj - 1), v.clamp(0, Hj - 1), dj)  # neighbor's 3D
            Xj = torch.where(inframe[:, None], Xj, Xi)                        # invalid -> fall back to own
            acc.append(Xj)
        P = torch.stack(acc, dim=0)                                          # [V,N,3]
        Xref = P.median(dim=0).values                                       # robust 3D consensus
        spread = (P - Xref[None]).norm(dim=2).mean(dim=0)                    # disagreement [N]
        zref = (torch.cat([Xref, torch.ones_like(Xref[:, :1])], 1) @ low[i].world_view_transform)[:, 2]
        scale = spread.median().clamp_min(1e-6)
        rel = torch.exp(-spread / scale)                                     # high = agree (low spread)
        D_ref.append(F.interpolate(zref.reshape(1, 1, Hl, Wl), size=(Hh, Ww), mode="bilinear",
                                   align_corners=True)[0, 0])
        REL.append(F.interpolate(rel.reshape(1, 1, Hl, Wl), size=(Hh, Ww), mode="bilinear",
                                 align_corners=True)[0, 0])
    return D_ref, REL


@torch.no_grad()
def generate_frgd_points(cam, image, D_ref, rel, grad, hole, base_mask=None,
                         tex_thr=0.05, hole_thr=0.5, rel_thr=0.5, max_points=20000):
    """Pick candidate pixels = under-reconstructed (hole>hole_thr) AND low-texture (grad<tex_thr, where 3DGS
    won't densify) AND reliable (rel>rel_thr), back-project via D_ref -> new 3D points + colors.
      image [3,H,W]; D_ref,rel,grad,hole [H,W]; hole = under-recon signal in [0,1] (e.g. 1-alpha or
      normalized |render_depth - D_ref|). Returns xyz [M,3], rgb [M,3] (M<=max_points)."""
    H, W = D_ref.shape
    cand = (grad < tex_thr) & (hole > hole_thr) & (rel > rel_thr) & (D_ref > 1e-6)
    if base_mask is not None:
        cand = cand & (base_mask > 0)
    idx = cand.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return image.new_zeros((0, 3)), image.new_zeros((0, 3))
    if idx.numel() > max_points:                                            # cap (avoid explosion)
        perm = torch.randperm(idx.numel(), device=idx.device)[:max_points]
        idx = idx[perm]
    vv = (idx // W).to(D_ref.dtype); uu = (idx % W).to(D_ref.dtype)
    z = D_ref.reshape(-1)[idx]
    xyz = unproject_pixels(cam, uu, vv, z)
    rgb = image.reshape(3, -1)[:, idx].t()
    return xyz, rgb
