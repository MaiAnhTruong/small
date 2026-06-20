#
# VS-Depth v5 — BDVR: Bidirectional Depth-Verified Refinement (the SUPPRESSION side).
#
# Rationale (DESIGN_AND_PROOF_v5.md):
#   FRGD (v4) only ADDS surface where it is missing (holes). It leaves the dominant sparse-view artifact
#   untouched: FLOATERS = Gaussians the *training* loss sustains (opacity>0, so 3DGS's own prune keeps them)
#   but that are NOT supported by multi-view geometric consensus -> they hurt held-out views.
#
#   BDVR is the DUAL of FRGD: withdraw capacity where the geometric support field is low. We compute a
#   per-Gaussian UNSUPPORTEDNESS phi in [0,1] and add a PERSISTENT prior  L_supp = supp_lambda * sum_i phi_i*o_i
#   to the loss (train.py). It competes with the photometric pull on opacity:
#     - true floater (phi high, weak photometric support) -> driven down -> pruned (low-opacity prune),
#     - genuine Gaussian wrongly flagged (strong photometric support) -> photometric pull wins -> RESTORED
#       (soft / reversible -> overcomes the irreversible-pruning danger of hard floater removal).
#
#   phi_i = (1 - GS_i) * OCC_i
#     GS_i  = exp(-d_i^2 / (2 r_s^2)),  d_i = dist(mu_i, nearest SfM anchor)         [anchor support]
#     OCC_i = (#views where mu_i is strictly IN FRONT of the rendered consensus surface) / (#in-frame views)
#   A genuine surface Gaussian is near an anchor (GS->1) OR lies on/behind the consensus in every view
#   (OCC->0) -> phi->0 (protected). A floater is far from anchors AND floats in front of the consensus
#   (the multi-view free-space violation signature) -> phi->1. Hard anchor protection: d_i < r_s => phi:=0.
#
#   NOTE the signal is purely GEOMETRIC (depth/position consensus + SfM anchors), never color variance:
#   color variance conflates "floater" with "texture" (the NAGC failure). This is the key correction.
#
import torch
from utils.covisibility import project_points, _sample


def nearest_anchor_dist(xyz, anchors, chunk=65536):
    """min Euclidean distance from each point in xyz [N,3] to the anchor set anchors [M,3] (chunked)."""
    if anchors is None or anchors.shape[0] == 0:
        return torch.full((xyz.shape[0],), float("inf"), device=xyz.device, dtype=xyz.dtype)
    out = torch.empty(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)
    for s in range(0, xyz.shape[0], chunk):
        e = min(s + chunk, xyz.shape[0])
        out[s:e] = torch.cdist(xyz[s:e], anchors).min(dim=1).values
    return out


@torch.no_grad()
def compute_supp_weight(xyz, cams, depths_z, anchors, tau=0.05, r_s=0.1, min_views=2):
    """Per-Gaussian unsupportedness phi in [0,1] (see module docstring).
      xyz [N,3] world positions; cams: list of cameras (need world_view_transform, FoVx/FoVy, W,H);
      depths_z[k]: [H,W] METRIC rendered depth (z, not inverse) of cam k; anchors [M,3] SfM anchor points.
    Returns phi [N]."""
    N = xyz.shape[0]
    dev, dt = xyz.device, xyz.dtype
    d = nearest_anchor_dist(xyz, anchors)
    GS = torch.exp(-(d * d) / (2.0 * float(r_s) ** 2))
    viol = torch.zeros(N, device=dev, dtype=dt)
    inframe = torch.zeros(N, device=dev, dtype=dt)
    for c, Dz in zip(cams, depths_z):
        u, v, z = project_points(xyz, c)
        W, H = int(c.image_width), int(c.image_height)
        infr = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 1e-6)
        Dj = _sample(Dz, u.clamp(0, W - 1), v.clamp(0, H - 1), W, H)
        viol += (infr & (z < Dj * (1.0 - float(tau)))).to(dt)              # mu_i strictly in front of surface
        inframe += infr.to(dt)
    OCC = torch.where(inframe >= float(min_views), viol / inframe.clamp_min(1.0), torch.zeros(N, device=dev, dtype=dt))
    phi = (1.0 - GS) * OCC
    phi = torch.where(d < float(r_s), torch.zeros(N, device=dev, dtype=dt), phi)   # hard anchor protection
    return phi.clamp(0.0, 1.0)
