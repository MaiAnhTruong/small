#
# VS-Depth: covisibility-gated depth supervision for sparse-view 3DGS.
#
# The base 3DGS depth loss is UNIFORM: |(invDepth - mono_invdepth) * depth_mask|.mean(), depth_mask=ones.
# VS-Depth replaces depth_mask with a per-pixel gate  w = g_cov(cov) * rel  that, by the bias-variance
# analysis (DESIGN_AND_PROOF.md, proven in tools/test_vsdepth_theory.py), is the MSE-optimal allocation:
#   - g_cov(cov): DECREASING in covisibility (#views observing a region). Low covisibility => low photometric
#     curvature H => high seed-variance => depth helps most. (EXACT geometric signal, not a learned proxy.)
#   - rel: DECREASING in mono-depth error proxy (depth-edge gradient). High error => more bias => less depth.
# gate_mode: "none"(no depth) / "uniform"(=base) / "covonly"(~CoMapGS, g_cov only) / "gated"(g_cov*rel)
#          / "fisher"(v2: weight = a*(H,delta), H = photometric FISHER information, see DESIGN_AND_PROOF_v2.md).
#
# v2 root insight: the curvature H in the theory = photometric Fisher info = MVS triangulation info
#   H(p) = Sum_j vis_j * (b_j f / z^2)^2 * |grad I_j|^2   (NOT the raw view count of CoMapGS, which drops
#   texture |grad I|^2 and geometry (b f/z^2)^2 -> wrong proxy of H; proven in tools/test_fisher_gate.py 8/8).
#
import torch
import torch.nn.functional as F
from utils.graphics_utils import fov2focal


def _intrinsics(cam):
    W, H = int(cam.image_width), int(cam.image_height)
    fx, fy = fov2focal(cam.FoVx, W), fov2focal(cam.FoVy, H)
    cx, cy = W / 2.0, H / 2.0
    return fx, fy, cx, cy, W, H


def project_points(xyz, cam):
    """world points [N,3] -> pixel (u,v) [N] and camera-z [N]. Row convention: cam_h = world_h @ wvt
    (matches scene/cameras.py world_view_transform, verified round-trip to 1e-14)."""
    fx, fy, cx, cy, W, H = _intrinsics(cam)
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    cam_h = torch.cat([xyz, ones], dim=1) @ cam.world_view_transform           # [N,4]
    z = cam_h[:, 2]
    zc = z.clamp_min(1e-8)
    u = fx * cam_h[:, 0] / zc + cx
    v = fy * cam_h[:, 1] / zc + cy
    return u, v, z


def unproject_depth(cam, depth_z):
    """rendered metric depth [H,W] -> world points [H,W,3]. Inverse of project_points."""
    fx, fy, cx, cy, W, H = _intrinsics(cam)
    dev, dt = depth_z.device, depth_z.dtype
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                            torch.arange(W, device=dev, dtype=dt), indexing="ij")
    x = (uu - cx) / fx * depth_z
    y = (vv - cy) / fy * depth_z
    cam_h = torch.stack([x, y, depth_z, torch.ones_like(depth_z)], dim=-1)     # [H,W,4]
    world_h = cam_h.reshape(-1, 4) @ torch.inverse(cam.world_view_transform)   # [HW,4]
    return world_h[:, :3].reshape(H, W, 3)


def _sample(img2d, u, v, W, H):
    """bilinear sample img2d [H,W] at pixel (u,v) [N] -> [N]."""
    gx = u / max(W - 1, 1) * 2 - 1
    gy = v / max(H - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    return F.grid_sample(img2d[None, None], grid, mode="bilinear",
                         padding_mode="border", align_corners=True).view(-1)


@torch.no_grad()
def covisibility_from_depths(cams, depths_z, tau=0.05):
    """cams: list of Camera; depths_z[i]: [H,W] METRIC depth of cam i (z, not inverse).
    Returns cov[i]: [H,W] = #other views that also observe cam i's surface point at each pixel
    (occlusion-checked by each other view's depth). EXACT given current geometry."""
    cov = []
    V = len(cams)
    for i in range(V):
        Hh, Ww = depths_z[i].shape
        Xi = unproject_depth(cams[i], depths_z[i]).reshape(-1, 3)              # [HW,3]
        c = torch.zeros(Hh * Ww, device=Xi.device, dtype=torch.float32)
        for j in range(V):
            if j == i:
                continue
            u, v, z = project_points(Xi, cams[j])
            Wj, Hj = int(cams[j].image_width), int(cams[j].image_height)
            inframe = (u >= 0) & (u < Wj) & (v >= 0) & (v < Hj) & (z > 1e-6)
            zj = _sample(depths_z[j], u.clamp(0, Wj - 1), v.clamp(0, Hj - 1), Wj, Hj)
            not_occluded = z <= zj * (1.0 + float(tau))                        # X at/in front of j's surface
            c += (inframe & not_occluded).float()
        cov.append(c.reshape(Hh, Ww))
    return cov


def reliability_from_invdepth(invdepth, sigma=0.10):
    """rel in (0,1]: LOW at mono-depth discontinuities (edges = unreliable mono-depth). invdepth [H,W]."""
    gy = torch.zeros_like(invdepth); gx = torch.zeros_like(invdepth)
    gy[1:, :] = invdepth[1:, :] - invdepth[:-1, :]
    gx[:, 1:] = invdepth[:, 1:] - invdepth[:, :-1]
    grad = (gx.abs() + gy.abs())
    s = grad.mean().clamp_min(1e-6) if sigma is None else float(sigma)
    return torch.exp(-grad / s)


def build_gate(mode, cov=None, invdepth=None, base_mask=None, gamma=1.0, rel_sigma=0.10):
    """Per-pixel depth-loss weight w [H,W], pinned to mean 1 (same total budget as uniform).
      none    -> zeros (no depth)
      uniform -> base_mask (= original 3DGS behavior)
      covonly -> g_cov(cov)             (~CoMapGS: covisibility gate, no reliability)
      gated   -> g_cov(cov) * rel       (ours: + mono-depth reliability)
    g_cov = 1/(1+cov)^gamma : decreasing in covisibility (MSE-optimal shape, Theorem P4)."""
    if mode == "none":
        return torch.zeros_like(base_mask) if base_mask is not None else torch.zeros_like(cov)
    if mode == "uniform":
        return base_mask if base_mask is not None else torch.ones_like(cov)
    assert cov is not None, "covonly/gated need covisibility"
    g_cov = 1.0 / (1.0 + cov).pow(float(gamma))
    w = g_cov
    if mode == "gated":
        assert invdepth is not None, "gated needs invdepth for reliability"
        w = w * reliability_from_invdepth(invdepth, rel_sigma)
    if base_mask is not None:
        w = w * base_mask                                                      # respect invalid/unreliable px
    w = w / w.mean().clamp_min(1e-8)                                           # pin mean -> 1 (budget kept)
    return w


class _CovCam:
    """Lightweight camera at reduced resolution for covisibility (FoV unchanged -> intrinsics scale with W,H)."""
    def __init__(self, cam, W, H):
        self.world_view_transform = cam.world_view_transform
        self.FoVx = cam.FoVx; self.FoVy = cam.FoVy
        self.image_width = W; self.image_height = H
        self.camera_center = cam.camera_center


@torch.no_grad()
def compute_gates(cams, depths_z, invdepths, base_masks, mode,
                  tau=0.05, gamma=1.0, rel_sigma=0.10, max_dim=200):
    """Per-camera full-res depth-loss gate [H,W]. Covisibility is computed at <=max_dim (it is smooth/low-freq)
    then the gate is upsampled to full res. depths_z[i]: full-res METRIC depth [H,W]; invdepths[i]: mono
    inverse depth [H,W] or None; base_masks[i]: [H,W]."""
    V = len(cams)
    low_cams, low_depths = [], []
    for i in range(V):
        H, W = depths_z[i].shape
        s = max(1, int(round(max(H, W) / float(max_dim))))
        Hl, Wl = max(2, H // s), max(2, W // s)
        low_cams.append(_CovCam(cams[i], Wl, Hl))
        low_depths.append(F.interpolate(depths_z[i][None, None], size=(Hl, Wl), mode="nearest")[0, 0])
    cov = covisibility_from_depths(low_cams, low_depths, tau)
    gates = []
    for i in range(V):
        H, W = depths_z[i].shape
        Hl, Wl = cov[i].shape
        invd_l = (F.interpolate(invdepths[i][None, None], size=(Hl, Wl), mode="bilinear", align_corners=True)[0, 0]
                  if invdepths[i] is not None else None)
        bm_l = F.interpolate(base_masks[i][None, None], size=(Hl, Wl), mode="nearest")[0, 0]
        m = mode if (mode != "gated" or invd_l is not None) else "covonly"     # no mono-depth -> covonly
        g_l = build_gate(m, cov=cov[i], invdepth=invd_l, base_mask=bm_l, gamma=gamma, rel_sigma=rel_sigma)
        gates.append(F.interpolate(g_l[None, None], size=(H, W), mode="bilinear", align_corners=True)[0, 0])
    return gates


# ============================ v2: Fisher-information gate ============================

def image_grad_mag(img):
    """|grad I| of luminance. img [C,H,W] or [H,W] in [0,1] -> [H,W] gradient magnitude (texture strength)."""
    g = img.mean(0) if img.dim() == 3 else img
    gx = torch.zeros_like(g); gy = torch.zeros_like(g)
    gx[:, 1:] = g[:, 1:] - g[:, :-1]
    gy[1:, :] = g[1:, :] - g[:-1, :]
    return gx.abs() + gy.abs()


@torch.no_grad()
def fisher_from_depths(cams, depths_z, grads, monoinv, tau=0.05):
    """Photometric FISHER information H and cross-view mono-depth disagreement delta, per cam.
    H[i](p) = Sum_{j!=i} vis_j * (b_j f_j / z_i(p)^2)^2 * |grad I_j(proj)|^2   (MVS triangulation info).
    delta[i](p) = RMS over covisible j of (mono_inv_j(proj) - mono_inv_i(p))   (None if no mono-depth).
    grads[i]: [H,W] |grad I_i| ; monoinv[i]: [H,W] aligned mono inverse depth or None."""
    Hs, Ds = [], []
    V = len(cams)
    for i in range(V):
        Hh, Ww = depths_z[i].shape
        zi = depths_z[i].reshape(-1).clamp_min(1e-6)                           # ref depth [N]
        Xi = unproject_depth(cams[i], depths_z[i]).reshape(-1, 3)
        Ci = cams[i].camera_center
        Hacc = torch.zeros_like(zi)
        mono_i = monoinv[i].reshape(-1) if monoinv[i] is not None else None
        dsum = torch.zeros_like(zi); dcnt = torch.zeros_like(zi)
        for j in range(V):
            if j == i:
                continue
            u, v, z = project_points(Xi, cams[j])
            Wj, Hj = int(cams[j].image_width), int(cams[j].image_height)
            inframe = (u >= 0) & (u < Wj) & (v >= 0) & (v < Hj) & (z > 1e-6)
            uc, vc = u.clamp(0, Wj - 1), v.clamp(0, Hj - 1)
            zj = _sample(depths_z[j], uc, vc, Wj, Hj)
            vis = (inframe & (z <= zj * (1.0 + float(tau)))).float()
            bj = torch.linalg.norm(Ci - cams[j].camera_center)                 # baseline (scalar)
            fj = fov2focal(cams[j].FoVx, Wj)
            geom = bj * fj / (zi * zi)                                         # disparity sensitivity [N]
            gj = _sample(grads[j], uc, vc, Wj, Hj)                             # |grad I_j| at projection
            Hacc = Hacc + vis * (geom * gj) ** 2
            if mono_i is not None and monoinv[j] is not None:
                mj = _sample(monoinv[j], uc, vc, Wj, Hj)
                dsum = dsum + vis * (mj - mono_i) ** 2
                dcnt = dcnt + vis
        Hs.append(Hacc.reshape(Hh, Ww))
        Ds.append((dsum / dcnt.clamp_min(1.0)).sqrt().reshape(Hh, Ww) if mono_i is not None else None)
    return Hs, Ds


def fisher_gate(H, delta, base_mask, c=0.5, floor=0.0, cap=8.0):
    """Weight = a*(H,delta) = c*Hn / (2*dn^2*Hn - c) in mean-normalized units (a*=cap when 2 dn^2 Hn <= c).
    DECREASING in H (low H = under-constrained -> MORE depth) and in delta (proven, P4). Mean-pinned to 1."""
    m = (base_mask > 0) if base_mask is not None else torch.ones_like(H, dtype=torch.bool)
    denom_mean = H[m].mean().clamp_min(1e-12) if m.any() else H.mean().clamp_min(1e-12)
    Hn = H / denom_mean
    if delta is not None:
        dmean = delta[m].mean().clamp_min(1e-12) if m.any() else delta.mean().clamp_min(1e-12)
        dn = delta / dmean
    else:
        dn = torch.ones_like(H)
    den = 2.0 * dn * dn * Hn - float(c)
    a = torch.where(den <= 0, torch.full_like(Hn, float(cap)),
                    (float(c) * Hn / den.clamp_min(1e-12)).clamp(float(floor), float(cap)))
    if base_mask is not None:
        a = a * base_mask
    return a / a.mean().clamp_min(1e-8)


@torch.no_grad()
def compute_fisher_gates(cams, depths_z, grads, invdepths, base_masks,
                         c=0.5, tau=0.05, max_dim=200, floor=0.0, cap=8.0):
    """Per-camera full-res Fisher gate [H,W] (computed at <=max_dim then upsampled; H is smooth/low-freq).
    Strict-gen: identical pipeline to compute_gates, only the per-pixel quantity is Fisher H (not count)."""
    V = len(cams)
    low_cams, lowd, lowg, lowm = [], [], [], []
    for i in range(V):
        H, W = depths_z[i].shape
        s = max(1, int(round(max(H, W) / float(max_dim))))
        Hl, Wl = max(2, H // s), max(2, W // s)
        low_cams.append(_CovCam(cams[i], Wl, Hl))
        lowd.append(F.interpolate(depths_z[i][None, None], size=(Hl, Wl), mode="nearest")[0, 0])
        lowg.append(F.interpolate(grads[i][None, None], size=(Hl, Wl), mode="bilinear", align_corners=True)[0, 0])
        lowm.append(F.interpolate(invdepths[i][None, None], size=(Hl, Wl), mode="bilinear", align_corners=True)[0, 0]
                    if invdepths[i] is not None else None)
    Hs, Ds = fisher_from_depths(low_cams, lowd, lowg, lowm, tau)
    gates = []
    for i in range(V):
        H, W = depths_z[i].shape
        bm_l = F.interpolate(base_masks[i][None, None], size=Hs[i].shape, mode="nearest")[0, 0]
        g_l = fisher_gate(Hs[i], Ds[i], bm_l, c=c, floor=floor, cap=cap)
        gates.append(F.interpolate(g_l[None, None], size=(H, W), mode="bilinear", align_corners=True)[0, 0])
    return gates
