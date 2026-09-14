#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_register.py

Geometry-only GLOBAL map-to-map registration: given two independently-captured
maps (e.g. two FAST-LIO2 sessions of a similar site, each zeroed at its own
launch pose), recover the single rigid transform T_RS that maps the SOURCE map
into the REFERENCE map's frame -- purely from geometry, no Scan Context, no
loop closure, no per-keyframe retrieval.

Pipeline (one solve on the FULL clouds):

    voxel downsample -> normals -> FPFH -> mutual-NN correspondences
      -> TEASER++ (full 6-DOF)  -> GICP refine (coarse -> fine)

Deliberately absent (and why), vs alignment_core.align_target's per-keyframe path:
  * NO floor-plane levelling. Outdoors there is no reliable floor; both clouds
    are already gravity-aligned by FAST-LIO. The Z offset between the two maps
    is a real unknown -- let TEASER solve it, don't assume floors coincide.
  * NO planar / Z constraint. constrain_planar() forces tilt->0 and z->0. The
    two FAST-LIO gravity frames disagree by a small real tilt (~2.4deg measured
    on palio2<->droneparkscan) and the launch altitudes differ; forcing them to
    zero injects that error. Solve full 6-DOF instead.
  * NO Scan Context windowing. Global registration works BECAUSE of large-scale
    distinctive structure; chopping the map into ~0.3s egocentric windows throws
    exactly that away.

Feature scales are OUTDOOR scale (voxel ~0.75m), driven by one knob
(AlignParams.GLOBAL_VOXEL); everything else is a multiple of it.

The solver internals (_fpfh / _correspondences / _teaser / _gicp) are imported
unchanged from alignment_core -- they are correct; only what is fed to them and
what is wrapped around them changes here. Kept as a SEPARATE file so the old
levelling/planar path in alignment_core stays available for comparison.

Usage:
    python3 global_register.py SRC.pcd REF.pcd
        SRC.pcd -- the map that gets aligned (e.g. palio2.pcd)
        REF.pcd -- the map defining the target frame (e.g. droneparkscan.pcd)
    prints T_RS (src->ref) and the health metrics.
"""
import sys

import numpy as np
import open3d as o3d

from alignment_core import (
    AlignParams, _fpfh, _correspondences, _teaser, _gicp,
    _yaw_tilt, _score_global as _coverage,
)


# =============================================================================
# the global registration
# =============================================================================
def register_global(src_pcd, ref_pcd, params, faiss_res=None, logger=None):
    """Geometry-only global registration, src -> ref, on the FULL maps.

    Returns (T_RS 4x4 mapping src->ref, info dict). Full 6-DOF: no levelling
    and no planar constraint -- TEASER++ solves rotation+translation from FPFH
    correspondences, GICP refines. Deterministic.
    """
    def log(m):
        if logger:
            logger(m)
    p = params
    v = p.GLOBAL_VOXEL

    # ---- 1. FPFH on the full clouds (NO level_cloud / NO _remove_ground) ----
    src_d, src_f = _fpfh(src_pcd, v, p.GLOBAL_NORMAL_R, p.GLOBAL_FEATURE_R)
    ref_d, ref_f = _fpfh(ref_pcd, v, p.GLOBAL_NORMAL_R, p.GLOBAL_FEATURE_R)
    log("FPFH: src=%d ref=%d pts (voxel=%.2f)"
        % (len(src_d.points), len(ref_d.points), v))

    # ---- 2. mutual-NN FPFH correspondences (src -> ref direction) ----
    src_c, ref_c = _correspondences(src_d, src_f, ref_d, ref_f, faiss_res)
    n_corr = src_c.shape[1]
    log("mutual correspondences: %d" % n_corr)
    if n_corr < 3:
        raise RuntimeError("Only %d correspondences (<3)." % n_corr)

    # ---- 3. TEASER++ (full 6-DOF, no constraint) ----
    T = _teaser(src_c, ref_c, p.GLOBAL_TEASER_NB, p)
    res = np.linalg.norm((T[:3, :3] @ src_c + T[:3, 3:4]) - ref_c, axis=0)
    n_inliers = int(np.count_nonzero(res < p.GLOBAL_TEASER_NB))
    yaw, tilt = _yaw_tilt(T[:3, :3])
    log("TEASER yaw=%.2f tilt=%.2f inliers=%d/%d (%.1f%%) t=[%.2f %.2f %.2f]"
        % (yaw, tilt, n_inliers, n_corr, 100.0 * n_inliers / max(n_corr, 1),
           T[0, 3], T[1, 3], T[2, 3]))

    # ---- 4. GICP refine, coarse -> fine (NO planar constraint) ----
    # T is already src->ref, the convention _gicp's init_T expects; it handles
    # the T_target_source inversion internally, so feed T straight back in.
    src_pts = np.ascontiguousarray(np.asarray(src_d.points), dtype=np.float64)
    ref_pts = np.ascontiguousarray(np.asarray(ref_d.points), dtype=np.float64)
    for dist in (v * 5.0, v * 2.0, v * 1.0):
        T = _gicp(src_pts, ref_pts, T, dist, p)
    yaw, tilt = _yaw_tilt(T[:3, :3])

    # ---- 5. coverage diagnostic + return ----
    cov, rmse = _coverage(src_pts, ref_pts, T, v * 2.0)
    log("post-GICP yaw=%.2f tilt=%.2f cov=%.2f rmse=%.2f t=[%.2f %.2f %.2f]"
        % (yaw, tilt, cov, rmse, T[0, 3], T[1, 3], T[2, 3]))

    info = {
        "n_correspondences": int(n_corr),
        "teaser_inliers": int(n_inliers),
        "yaw_deg": yaw,
        "tilt_deg": tilt,
        "coverage": cov,
        "inlier_rmse": rmse,
        "tx": float(T[0, 3]), "ty": float(T[1, 3]), "tz": float(T[2, 3]),
    }
    return T, info


# =============================================================================
# CLI
# =============================================================================
def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python3 global_register.py SRC.pcd REF.pcd\n"
            "  SRC.pcd -- map that gets aligned (e.g. palio2.pcd)\n"
            "  REF.pcd -- map defining the target frame (e.g. droneparkscan.pcd)")

    src_path, ref_path = sys.argv[1], sys.argv[2]
    src = o3d.io.read_point_cloud(src_path)
    ref = o3d.io.read_point_cloud(ref_path)
    if len(src.points) == 0 or len(ref.points) == 0:
        raise SystemExit("empty cloud: src=%d ref=%d pts"
                         % (len(src.points), len(ref.points)))

    T, info = register_global(src, ref, AlignParams(), logger=print)
    print("\nT_RS (src->ref) =\n%s" % np.array2string(T, precision=3, suppress_small=True))
    print("\ninfo = %s" % info)

    # health read (see the pipeline doc): cross-scale agreement is the strongest
    # signal -- re-run with a different GLOBAL_VOXEL and confirm the transform
    # matches. Ignore the inlier PERCENTAGE; >95%% outliers is normal for TEASER.
    ok = info["coverage"] >= 0.5 and info["teaser_inliers"] >= 20
    print("\nsanity: coverage=%.2f inliers=%d -> %s"
          % (info["coverage"], info["teaser_inliers"],
             "looks aligned" if ok else "SUSPECT -- check overlap / voxel scale"))


if __name__ == "__main__":
    main()
