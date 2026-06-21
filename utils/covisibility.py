#
# VS-Depth geometry primitives (projection / unprojection / multi-view sampling) shared by FRGD (refine_depth_maps,
# generate_frgd_points) and DIGS (dense depth init). Round-trip verified (test_frgd). The earlier depth-LOSS
# gating (covonly/gated/fisher) was measured dead and has been removed; only these primitives remain.
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


class _CovCam:
    """Lightweight camera at reduced resolution for multi-view fusion (FoV unchanged -> intrinsics scale with W,H)."""
    def __init__(self, cam, W, H):
        self.world_view_transform = cam.world_view_transform
        self.FoVx = cam.FoVx; self.FoVy = cam.FoVy
        self.image_width = W; self.image_height = H
        self.camera_center = cam.camera_center


def image_grad_mag(img):
    """|grad I| of luminance. img [C,H,W] or [H,W] in [0,1] -> [H,W] gradient magnitude (texture strength)."""
    g = img.mean(0) if img.dim() == 3 else img
    gx = torch.zeros_like(g); gy = torch.zeros_like(g)
    gx[:, 1:] = g[:, 1:] - g[:, :-1]
    gy[1:, :] = g[1:, :] - g[:-1, :]
    return gx.abs() + gy.abs()
