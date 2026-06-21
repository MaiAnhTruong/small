#
# VS-Depth v6 — FRGD-G: geometry-correct INITIALIZATION of densified Gaussians (DESIGN_AND_PROOF_v6.md).
#
# FRGD picks WHERE to add (refined depth, holes). FRGD-G fixes the SHAPE of each added Gaussian:
#   - lateral world std  sigma_lat = c_f * z / f         (Eq 2.1: the pixel-frustum footprint at depth z)
#   - along-ray std      sigma_n   = beta * sigma_lat    (Eq 3.1: flattened disk, beta=0.25, non-degenerate)
#   - orientation: thin axis ALONG the view ray (camera-facing) -> minimal depth-spread -> minimal NVS haze.
# Opacity & placement are untouched (handled by add_frgd_points / frgd_step) so the proven +0.302 is kept.
#
# Proven on CPU (tools/test_frgd_g.py 13/13): T2 footprint z/f & distCUDA2 over-size; T3 disk cuts off-surface
# (haze) mass 13.6x; T1 the init survives optimization in the sparse+few-iter regime.
#
import torch
import torch.nn.functional as F
from utils.graphics_utils import fov2focal
from utils.covisibility import project_points


def matrix_to_quaternion(R):
    """Batched rotation matrix [M,3,3] -> unit quaternion (w,x,y,z) matching utils.general_utils.build_rotation
    (i.e. build_rotation(matrix_to_quaternion(R)) == R). Stable Shepperd 4-case selection."""
    m = R
    eps = 1e-12
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    s0 = torch.sqrt((t + 1.0).clamp_min(eps)) * 2.0                       # = 4w
    q0 = torch.stack([0.25 * s0, (m[:, 2, 1] - m[:, 1, 2]) / s0,
                      (m[:, 0, 2] - m[:, 2, 0]) / s0, (m[:, 1, 0] - m[:, 0, 1]) / s0], dim=1)
    s1 = torch.sqrt((1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp_min(eps)) * 2.0   # = 4x
    q1 = torch.stack([(m[:, 2, 1] - m[:, 1, 2]) / s1, 0.25 * s1,
                      (m[:, 0, 1] + m[:, 1, 0]) / s1, (m[:, 0, 2] + m[:, 2, 0]) / s1], dim=1)
    s2 = torch.sqrt((1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]).clamp_min(eps)) * 2.0   # = 4y
    q2 = torch.stack([(m[:, 0, 2] - m[:, 2, 0]) / s2, (m[:, 0, 1] + m[:, 1, 0]) / s2,
                      0.25 * s2, (m[:, 1, 2] + m[:, 2, 1]) / s2], dim=1)
    s3 = torch.sqrt((1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]).clamp_min(eps)) * 2.0   # = 4z
    q3 = torch.stack([(m[:, 1, 0] - m[:, 0, 1]) / s3, (m[:, 0, 2] + m[:, 2, 0]) / s3,
                      (m[:, 1, 2] + m[:, 2, 1]) / s3, 0.25 * s3], dim=1)
    c0 = t > 0
    c1 = (~c0) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    c2 = (~c0) & (~c1) & (m[:, 1, 1] >= m[:, 2, 2])
    q = torch.where(c0[:, None], q0, torch.where(c1[:, None], q1, torch.where(c2[:, None], q2, q3)))
    return F.normalize(q, dim=1)


@torch.no_grad()
def frgd_g_shape(xyz, cam, c_f=1.0, beta=0.25, sigma_max_frac=0.1, extent=1.0):
    """Per-point geometry-correct scale + rotation for densified Gaussians (camera-facing disk).
      xyz [M,3] world centers; cam: camera (FoVx, image_width, world_view_transform, camera_center).
    Returns scales [M,3] world std (axes = [tangent, tangent, ray]) and quats [M,4] (w,x,y,z)."""
    f = fov2focal(cam.FoVx, int(cam.image_width))
    _, _, z = project_points(xyz, cam)                                   # z = camera-space depth of each point
    z = z.clamp_min(1e-6)
    sig_lat = (c_f * z / f).clamp(1e-6, float(sigma_max_frac) * float(extent))
    sig_n = (float(beta) * sig_lat)
    scales = torch.stack([sig_lat, sig_lat, sig_n], dim=1)               # axis 2 (ray) is the thin one

    cc = cam.camera_center.to(xyz.device, xyz.dtype)
    r = F.normalize(xyz - cc[None], dim=1)                               # view ray (world), the thin axis
    up = torch.tensor([0.0, 1.0, 0.0], device=xyz.device, dtype=xyz.dtype).expand_as(r)
    alt = torch.tensor([1.0, 0.0, 0.0], device=xyz.device, dtype=xyz.dtype).expand_as(r)
    ref = torch.where(((r * up).sum(1).abs() > 0.99)[:, None], alt, up)  # avoid degenerate cross product
    t1 = F.normalize(torch.cross(ref, r, dim=1), dim=1)
    t2 = torch.cross(r, t1, dim=1)
    R = torch.stack([t1, t2, r], dim=2)                                  # columns = scale axes [tangent,tangent,ray]
    quats = matrix_to_quaternion(R)
    return scales, quats
