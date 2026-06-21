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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        # ---- VS-Depth: depth supervision mode ----
        #   none    : no depth supervision (= vanilla 3DGS)
        #   uniform : 3DGS depth loss with depth_mask (the proven setting; used with FRGD/DIGS densification)
        self.gate_mode = "uniform"
        self.cov_tau = 0.05            # depth tolerance for multi-view fusion (refine_depth_maps)
        self.cov_max_dim = 200         # multi-view fusion computed at this max image dim (smooth -> downsampled)
        # ---- VS-Depth v4: FRGD (Fisher-Reliability-Guided Densification) ----
        # densify depth into PLACEMENT (non-zero-sum) instead of LOSS (zero-sum, measured dead). Seeds new
        # Gaussians at multi-view-refined depth in under-reconstructed + low-texture + reliable regions.
        # densify_mode (ablation ladder, all on top of base 3DGS densify):
        #   none       -> base 3DGS densify only (bit-identical baseline)
        #   rawdensify -> naive raw-mono depth densify (no refine / no reliability / no texture gate)
        #   frgd       -> FRGD (v4): multi-view-REFINED depth placement + reliability filter + texture targeting
        #   frgdg      -> FRGD-G (v6): frgd placement + geometry-correct SHAPE init (frustum z/f anisotropic disk,
        #                 camera-facing); opacity unchanged (proofs test_frgd_g 13/13)
        #   cgd        -> CGD (v7, NEWEST): frgdg + Confidence-guided OPACITY init  o_init = 0.1 * conf, where
        #                 conf = multi-view fusion reliability (rel). Depth-uncertain points are born faint and
        #                 self-prune unless photometric evidence rescues them (proofs test_cgd 11/11).
        # Typically run WITH gate_mode=uniform (-d depths): uniform depth loss (proven) + densification.
        self.densify_mode = "none"
        self.frgd_start = 2000         # begin FRGD after geometry roughly forms
        self.frgd_interval = 1000      # run FRGD every N iters (renders all train depths + refine)
        self.frgd_max_per_step = 30000 # cap new points per FRGD step (after dedup)
        self.frgd_tex_thr = 0.02       # low-texture: |grad I| below this (where 3DGS densify fails)
        self.frgd_hole_thr = 0.15      # under-recon: (render_z - D_ref)/D_ref above this = hole behind prior surface
        self.frgd_rel_thr = 0.5        # reliability: multi-view agreement of D_ref above this = trust placement
        # ---- VS-Depth v6/v7: FRGD-G shape init + CGD confidence-opacity (active for densify_mode frgdg/cgd) ----
        self.frgdg_cf = 1.0            # lateral world std = frgdg_cf * z / f  (pixel-frustum footprint, Eq 2.1)
        self.frgdg_beta = 0.25         # along-ray std = frgdg_beta * lateral std (flattened disk, Eq 3.1)
        self.frgdg_sigma_max_frac = 0.1 # clamp lateral std <= frgdg_sigma_max_frac * cameras_extent (safety)
        self.cgd_rel_floor = 0.1       # cgd: keep candidates with rel>this (graded opacity does the rest), then
                                       #      o_init = 0.1 * rel  -> low-rel points faint -> self-prune (Eq 4.1/5.1)
        # ---- VS-Depth v8: DIGS dense depth-backprojected INITIALIZATION (init_mode="depth") ----
        # Inject the depth capacity at iter 0 (dense, geometry-correct surfels) instead of mid-training, so it
        # gets full optimization budget (init-persistence). See DESIGN_AND_PROOF_v8 (proofs test_digs 9/9).
        self.init_mode = "sfm"         # "sfm" (base: SfM-only init) | "depth" (DIGS: + dense depth init at iter 0)
        self.digs_stride = 4           # pixel subsample stride for back-projection (controls init density)
        self.digs_rel_floor = 0.3      # keep back-projected pixels with multi-view reliability rel > this
        self.digs_max_points = 1500000 # cap total dense-init points (after dedup vs SfM)
        self.digs_conf_opacity = False # True: o_init=0.1*rel (CGD-style); default constant 0.1 (robust)
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
