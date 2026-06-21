#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from utils.covisibility import image_grad_mag
from utils.frgd import refine_depth_maps, generate_frgd_points, unproject_pixels
from utils.frgd_g import frgd_g_shape
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def _frgd_dedup(X, existing, voxel):
    """Drop new points whose voxel already contains an existing Gaussian (avoid redundant overlap)."""
    if voxel <= 0 or X.shape[0] == 0:
        return torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    def key(P):
        q = torch.floor(P / voxel).long()
        return (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    exk = key(existing).unique()
    return ~torch.isin(key(X), exk)


@torch.no_grad()
def frgd_step(scene, gaussians, pipe, background, opt, separate_sh, use_exp, iteration):
    """VS-Depth v4 FRGD: render all train depths, refine mono-depth across views (D_ref), then SEED new
    Gaussians at D_ref in under-reconstructed (render farther than D_ref) + low-texture + reliable regions."""
    cams = scene.getTrainCameras()
    mono_z, imgs, grads, masks, render_z = [], [], [], [], []
    any_depth = False
    for c in cams:
        pkg = render(c, gaussians, pipe, background, use_trained_exp=use_exp, separate_sh=separate_sh)
        inv_r = pkg["depth"][0].clamp_min(1e-6)                              # rendered inverse depth
        render_z.append(1.0 / inv_r)
        if c.invdepthmap is not None and c.depth_reliable:
            mono_z.append(1.0 / c.invdepthmap[0].clamp_min(1e-6)); any_depth = True
        else:
            mono_z.append(1.0 / inv_r)
        if not hasattr(c, "_vsd_grad"):
            c._vsd_grad = image_grad_mag(c.original_image.cuda())
        grads.append(c._vsd_grad); imgs.append(c.original_image.cuda())
        masks.append(c.depth_mask[0] if getattr(c, "depth_mask", None) is not None else torch.ones_like(inv_r))
    if not any_depth:
        return
    mode = opt.densify_mode
    shape_mode = mode in ("frgdg", "cgd")   # FRGD-G geometry-correct shape (frustum z/f anisotropic disk)
    if mode == "rawdensify":          # ABLATION: naive raw-mono densify (NO refine, NO reliability/texture gate)
        place = mono_z; REL = [torch.ones_like(m) for m in mono_z]; tex_thr = 1e9
    else:                             # frgd / frgdg / cgd: multi-view-REFINED depth + reliability + texture target
        place, REL = refine_depth_maps(cams, mono_z, tau=opt.cov_tau, max_dim=opt.cov_max_dim)
        tex_thr = opt.frgd_tex_thr
    rel_thr = opt.cgd_rel_floor if mode == "cgd" else opt.frgd_rel_thr   # cgd: low floor, opacity does the rest
    Xs, Cs, Ss, Qs, Os = [], [], [], [], []
    for i in range(len(cams)):
        hole = ((render_z[i] - place[i]) / place[i].clamp_min(1e-6)).clamp(0.0, 10.0)   # render behind prior surface
        out = generate_frgd_points(cams[i], imgs[i], place[i], REL[i], grads[i], hole, base_mask=masks[i],
                                   tex_thr=tex_thr, hole_thr=opt.frgd_hole_thr, rel_thr=rel_thr,
                                   max_points=opt.frgd_max_per_step, return_rel=(mode == "cgd"))
        xyz, rgb = (out[0], out[1])
        if xyz.shape[0] == 0:
            continue
        Xs.append(xyz); Cs.append(rgb)
        if shape_mode:
            sc, qu = frgd_g_shape(xyz, cams[i], c_f=opt.frgdg_cf, beta=opt.frgdg_beta,
                                  sigma_max_frac=opt.frgdg_sigma_max_frac, extent=scene.cameras_extent)
            Ss.append(sc); Qs.append(qu)
        if mode == "cgd":
            Os.append((0.1 * out[2]).clamp(1e-4, 0.99))                  # o_init = 0.1 * conf (Eq 4.1)
    if not Xs:
        print(f"[{mode} {iteration}] 0 candidates", flush=True); return
    X = torch.cat(Xs, 0).cuda(); C = torch.cat(Cs, 0).cuda()
    S = torch.cat(Ss, 0).cuda() if shape_mode else None
    Q = torch.cat(Qs, 0).cuda() if shape_mode else None
    O = torch.cat(Os, 0).cuda() if mode == "cgd" else None
    keep = _frgd_dedup(X, gaussians.get_xyz, opt.percent_dense * scene.cameras_extent)
    X, C = X[keep], C[keep]
    if S is not None: S, Q = S[keep], Q[keep]
    if O is not None: O = O[keep]
    if X.shape[0] > opt.frgd_max_per_step:
        idx = torch.randperm(X.shape[0], device=X.device)[:opt.frgd_max_per_step]
        X, C = X[idx], C[idx]
        if S is not None: S, Q = S[idx], Q[idx]
        if O is not None: O = O[idx]
    n = gaussians.add_frgd_points(X, C, scales=S, quats=Q, opacities=O)
    print(f"[{mode} {iteration}] +{n} pts -> {gaussians.get_xyz.shape[0]} total", flush=True)


@torch.no_grad()
def digs_init(scene, gaussians, opt):
    """VS-Depth v8 DIGS: dense depth-backprojected INITIALIZATION at iter 0 (DESIGN_AND_PROOF_v8, proofs 9/9).
    Back-project the multi-view-refined aligned depth of all train views (subsampled, reliability>floor) as
    geometry-correct surfels (frustum z/f disk, FRGD-G) and ADD them to the SfM cloud BEFORE training, so they
    receive the full optimization budget (init-persistence) instead of the partial budget of mid-training FRGD."""
    cams = scene.getTrainCameras()
    mono_z, vcams = [], []
    for c in cams:
        if c.invdepthmap is not None and c.depth_reliable:
            mono_z.append(1.0 / c.invdepthmap[0].clamp_min(1e-6)); vcams.append(c)
    if not mono_z:
        print("[digs] no reliable depth -> skip dense init", flush=True); return
    D_ref, REL = refine_depth_maps(vcams, mono_z, tau=opt.cov_tau, max_dim=opt.cov_max_dim)
    s = max(1, int(opt.digs_stride))
    Xs, Cs, Ss, Qs, Os = [], [], [], [], []
    for i, c in enumerate(vcams):
        H, W = D_ref[i].shape
        vv, uu = torch.meshgrid(torch.arange(0, H, s, device=D_ref[i].device),
                                torch.arange(0, W, s, device=D_ref[i].device), indexing="ij")
        vv, uu = vv.reshape(-1), uu.reshape(-1)
        z = D_ref[i][vv, uu]; rel = REL[i][vv, uu]
        keep = (rel > opt.digs_rel_floor) & (z > 1e-6)
        if keep.sum() == 0:
            continue
        vv, uu, z, rel = vv[keep], uu[keep], z[keep], rel[keep]
        xyz = unproject_pixels(c, uu.float(), vv.float(), z)
        rgb = c.original_image.cuda()[:, vv, uu].t()                          # color at the back-projected pixels
        sc, qu = frgd_g_shape(xyz, c, c_f=opt.frgdg_cf, beta=opt.frgdg_beta,
                              sigma_max_frac=opt.frgdg_sigma_max_frac, extent=scene.cameras_extent)
        Xs.append(xyz); Cs.append(rgb); Ss.append(sc); Qs.append(qu)
        if opt.digs_conf_opacity:
            Os.append((0.1 * rel).clamp(1e-4, 0.99))                          # CGD-style (optional); default 0.1
    if not Xs:
        print("[digs] 0 candidates -> skip dense init", flush=True); return
    X = torch.cat(Xs).cuda(); C = torch.cat(Cs).cuda(); S = torch.cat(Ss).cuda(); Q = torch.cat(Qs).cuda()
    O = torch.cat(Os).cuda() if opt.digs_conf_opacity else None
    keep = _frgd_dedup(X, gaussians.get_xyz, opt.percent_dense * scene.cameras_extent)   # drop overlap with SfM
    X, C, S, Q = X[keep], C[keep], S[keep], Q[keep]
    if O is not None: O = O[keep]
    if X.shape[0] > opt.digs_max_points:
        idx = torch.randperm(X.shape[0], device=X.device)[:opt.digs_max_points]
        X, C, S, Q = X[idx], C[idx], S[idx], Q[idx]
        if O is not None: O = O[idx]
    n = gaussians.add_frgd_points(X, C, scales=S, quats=Q, opacities=O)
    print(f"[digs] dense depth init: +{n} pts -> {gaussians.get_xyz.shape[0]} total", flush=True)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if opt.init_mode == "depth" and not checkpoint:        # DIGS: dense depth init at iter 0 (v8)
        digs_init(scene, gaussians, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization (uniform 3DGS depth loss; gate_mode none disables it)
        Ll1depth_pure = 0.0
        if opt.gate_mode != "none" and depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_w = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_w).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # VS-Depth depth-guided densification (rawdensify=naive; frgd=refined+targeted;
                # frgdg=+geometry-correct shape; cgd=+confidence-opacity). The ADD side.
                if opt.densify_mode in ("frgd", "rawdensify", "frgdg", "cgd") and iteration >= opt.frgd_start and iteration % opt.frgd_interval == 0:
                    frgd_step(scene, gaussians, pipe, background, opt, SPARSE_ADAM_AVAILABLE,
                              dataset.train_test_exp, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
