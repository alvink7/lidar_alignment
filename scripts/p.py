#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alignment_core.py
-----------------
Core point-cloud alignment pipeline, extracted verbatim (in behaviour) from the
multibag_mapping notebook so the live ROS node and the offline reference-prep
script share ONE implementation.

Pipeline per alignment:
    level (ground plane -> Z up, ground at 0)
      -> remove ground points
      -> normalize XY to origin
      -> FPFH features
      -> FAISS GPU mutual-NN correspondences
      -> TEASER++ global registration
      -> small_gicp GICP refinement (coarse then fine)
    => 4x4 transform mapping TARGET frame into REFERENCE frame.

This module is pure Python / numpy / open3d / faiss / teaserpp_python /
small_gicp.  It imports NOTHING from rospy, so it can be unit-tested off-robot.
"""

import numpy as np
import open3d as o3d
import copy

import teaserpp_python
import faiss
import small_gicp


# =============================================================================
# Parameters (defaults copied from the notebook config cell).
# The ROS node overrides these from the parameter server; the offline prep
# script imports them directly.
# =============================================================================
class AlignParams(object):
    ICP_VOXEL    = 0.10
    RANSAC_VOXEL = 0.30
    VOXEL_SIZE   = 0.05

    GROUND_DIST_THRESHOLD = 0.15
    GROUND_REMOVE_Z       = 0.30

    NORMAL_RADIUS_R  = RANSAC_VOXEL * 2
    FEATURE_RADIUS_R = RANSAC_VOXEL * 5

    TEASER_NOISE_BOUND = RANSAC_VOXEL * 2.0
    TEASER_GNC_FACTOR  = 1.4
    TEASER_MAX_ITER    = 100

    ICP_DIST_COARSE = 0.5
    ICP_DIST_FINE   = 0.1
    ICP_MAX_ITER    = 50
    NORMAL_RADIUS   = ICP_VOXEL * 2

    GROUND_PLANE_ITERS = 200
    # Ground-plane search: reject planes tilted more than this from horizontal
    # (so walls/ceilings aren't mistaken for the floor indoors), retrying up to
    # GROUND_PLANE_ATTEMPTS times, removing found inliers each try.
    GROUND_MAX_TILT_DEG = 30.0
    GROUND_PLANE_ATTEMPTS = 5


# =============================================================================
# Ground extraction / levelling   (Cell 10)
# =============================================================================
def _find_horizontal_plane(pcd, dist_threshold, plane_iters, params, label=""):
    """Find the GROUND plane, not just the biggest plane. Indoors the largest
    plane is often a wall; this rejects planes whose normal isn't near-vertical
    and keeps searching (removing found inliers) until a horizontal plane is
    found. Returns (normal, inlier_indices) or falls back to the biggest plane
    if no horizontal one is found."""
    max_tilt = getattr(params, "GROUND_MAX_TILT_DEG", 30.0)
    work = pcd
    remaining_idx = np.arange(len(pcd.points))
    best_fallback = None
    for attempt in range(getattr(params, "GROUND_PLANE_ATTEMPTS", 5)):
        if len(work.points) < 3:
            break
        pm, inl = work.segment_plane(
            distance_threshold=dist_threshold, ransac_n=3,
            num_iterations=plane_iters)
        n = np.array(pm[:3], dtype=np.float64)
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        tilt = np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))
        # map local inliers back to original indices
        global_inl = remaining_idx[inl]
        if best_fallback is None or len(inl) > len(best_fallback[1]):
            best_fallback = (n, global_inl)
        if tilt <= max_tilt:
            return n, global_inl        # found the ground
        # not horizontal: remove these inliers and search the rest
        mask = np.ones(len(work.points), dtype=bool)
        mask[inl] = False
        remaining_idx = remaining_idx[mask]
        work = work.select_by_index(np.where(mask)[0])
    # no horizontal plane found; use the largest we saw
    return best_fallback


def extract_ground_and_level(pcd, dist_threshold, plane_iters, label="",
                            params=None):
    """Fit ground plane; return 4x4 T_level that rotates cloud upright and puts
    the ground at Z=0. Uses a horizontal-constrained search so walls are not
    mistaken for the floor (critical indoors)."""
    if params is None:
        params = AlignParams()
    normal, inliers = _find_horizontal_plane(
        pcd, dist_threshold, plane_iters, params, label)

    z_axis = np.array([0.0, 0.0, 1.0])
    axis   = np.cross(normal, z_axis)
    angle  = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))

    if np.linalg.norm(axis) < 1e-6:
        R = np.eye(3)
    else:
        axis /= np.linalg.norm(axis)
        K = np.array([
            [ 0,       -axis[2],  axis[1]],
            [ axis[2],  0,       -axis[0]],
            [-axis[1],  axis[0],  0      ],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    T_level = np.eye(4)
    T_level[:3, :3] = R
    pts_rot_z = (np.asarray(pcd.points) @ R.T)[:, 2]
    ground_z = np.percentile(pts_rot_z, 2)
    T_level[2, 3] = -ground_z
    return T_level


def apply_transform(pcd, T):
    out = copy.deepcopy(pcd)
    out.transform(T)
    return out


# =============================================================================
# Ground removal / XY normalize / FPFH   (Cell 13)
# =============================================================================
def remove_ground_points(pcd, z_threshold):
    pts  = np.asarray(pcd.points)
    mask = pts[:, 2] > z_threshold
    above = o3d.geometry.PointCloud()
    above.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_normals():
        above.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    if pcd.has_colors():
        above.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return above


def normalize_xy(pcd):
    ctr = pcd.get_axis_aligned_bounding_box().get_center()
    T = np.eye(4)
    T[0, 3] = -ctr[0]
    T[1, 3] = -ctr[1]
    return apply_transform(pcd, T), ctr


def preprocess_for_ransac(pcd, voxel, normal_r, feature_r):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_r, max_nn=100),
    )
    return down, fpfh


def make_T(tx=0, ty=0, tz=0):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
    return T


# =============================================================================
# FAISS GPU correspondences   (Cell 13)
# =============================================================================
def _faiss_supports_gpu():
    """True if this faiss build has GPU support AND a GPU is visible."""
    try:
        return (hasattr(faiss, "StandardGpuResources")
                and faiss.get_num_gpus() > 0)
    except Exception:
        return False


def get_correspondences_faiss(src_fpfh, tgt_fpfh, src_pcd, tgt_pcd,
                              faiss_res=None, mutual=True):
    """Mutual NN FPFH matching. Returns (src_corr, tgt_corr) as (3,N).
    Uses the GPU if this faiss build supports it and a GPU is present;
    otherwise falls back to a CPU IndexFlatL2. Pass a persistent
    faiss.StandardGpuResources() in faiss_res to avoid re-allocating GPU
    scratch each call (ignored on CPU)."""
    src_feat = np.ascontiguousarray(np.array(src_fpfh.data).T, dtype=np.float32)
    tgt_feat = np.ascontiguousarray(np.array(tgt_fpfh.data).T, dtype=np.float32)
    src_pts  = np.asarray(src_pcd.points)
    tgt_pts  = np.asarray(tgt_pcd.points)

    d = src_feat.shape[1]
    if _faiss_supports_gpu():
        if faiss_res is None:
            faiss_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(faiss_res, 0, faiss.IndexFlatL2(d))
    else:
        index = faiss.IndexFlatL2(d)   # CPU fallback

    index.add(tgt_feat)
    _, fwd = index.search(src_feat, 1)
    fwd = fwd.flatten()

    if mutual:
        index.reset()
        index.add(src_feat)
        _, bwd = index.search(tgt_feat[fwd], 1)
        bwd = bwd.flatten()
        mask = (bwd == np.arange(len(fwd)))
        return src_pts[mask].T, tgt_pts[fwd[mask]].T

    return src_pts.T, tgt_pts[fwd].T


# =============================================================================
# TEASER++   (Cell 13)
# =============================================================================
class TEASERResult(object):
    def __init__(self, R, t, src_corr, tgt_corr, noise_bound):
        self.transformation = np.eye(4)
        self.transformation[:3, :3] = R
        self.transformation[:3, 3]  = t
        src_t   = (R @ src_corr).T + t
        dists   = np.linalg.norm(src_t - tgt_corr.T, axis=1)
        inliers = dists < noise_bound
        self.fitness     = float(inliers.mean()) if len(dists) else 0.0
        self.inlier_rmse = (float(np.sqrt((dists[inliers] ** 2).mean()))
                            if inliers.any() else 0.0)


def run_teaser(src_corr, tgt_corr, noise_bound, gnc_factor, max_iter):
    p = teaserpp_python.RobustRegistrationSolver.Params()
    p.cbar2                         = 1.0
    p.noise_bound                   = noise_bound
    p.estimate_scaling              = False
    p.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver
        .ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
    p.rotation_gnc_factor           = gnc_factor
    p.rotation_max_iterations       = max_iter
    p.rotation_cost_threshold       = 1e-12
    solver = teaserpp_python.RobustRegistrationSolver(p)
    solver.solve(src_corr, tgt_corr)
    sol = solver.getSolution()
    return sol.rotation, sol.translation


# =============================================================================
# small_gicp ICP   (Cell 15)
# =============================================================================
class _ICPResult(object):
    def __init__(self, T, fitness, rmse):
        self.transformation = T
        self.fitness        = fitness
        self.inlier_rmse    = rmse


def icp_pass(src, tgt, initial_transform, dist_threshold, max_iter):
    src_pts = np.ascontiguousarray(np.asarray(src.points), dtype=np.float64)
    tgt_pts = np.ascontiguousarray(np.asarray(tgt.points), dtype=np.float64)
    src_sg = small_gicp.PointCloud(src_pts)
    tgt_sg = small_gicp.PointCloud(tgt_pts)
    res = small_gicp.align(
        tgt_sg, src_sg,
        init_T_target_source=np.linalg.inv(initial_transform),
        registration_type="GICP",
        max_correspondence_distance=dist_threshold,
        num_threads=8,
        max_iterations=max_iter,
    )
    T = np.linalg.inv(res.T_target_source)
    n_in = res.num_inliers
    fitness = n_in / max(len(src_pts), 1)
    rmse = float(np.sqrt(res.error / n_in)) if n_in > 0 else float("inf")
    return _ICPResult(T, fitness, rmse)


# =============================================================================
# ICP-voxel downsample + normals   (notebook Cell 9)
# The notebook levels and aligns the ICP_VOXEL-downsampled clouds, NOT the raw
# clouds. Skipping this step makes segment_plane unreliable on dense raw data
# and breaks the alignment, so it must happen before levelling.
# =============================================================================
def downsample_for_icp(pcd, params):
    down = pcd.voxel_down_sample(params.ICP_VOXEL)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.NORMAL_RADIUS, max_nn=30))
    return down


# =============================================================================
# Reference preprocessing  -- run ONCE, cached by the node / prep script
# =============================================================================
class PreparedReference(object):
    """Everything about the reference that the per-target alignment needs,
    computed once."""
    def __init__(self, ref_cloud, params):
        self.params = params
        # Cell 9: ICP-voxel downsample + normals BEFORE levelling
        ref_icp = downsample_for_icp(ref_cloud, params)
        # Cell 10: level the downsampled cloud
        self.T_ref_level = extract_ground_and_level(
            ref_icp, params.GROUND_DIST_THRESHOLD,
            params.GROUND_PLANE_ITERS, label="REF", params=params)
        self.ref_levelled = apply_transform(ref_icp, self.T_ref_level)
        self.ref_levelled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.NORMAL_RADIUS, max_nn=30))
        # ground-removed / normalized / FPFH for global registration
        ref_above = remove_ground_points(self.ref_levelled, params.GROUND_REMOVE_Z)
        ref_norm, self.ref_xy_ctr = normalize_xy(ref_above)
        self.ref_down, self.ref_fpfh = preprocess_for_ransac(
            ref_norm, params.RANSAC_VOXEL,
            params.NORMAL_RADIUS_R, params.FEATURE_RADIUS_R)


# =============================================================================
# Full target alignment  ->  T mapping target frame into reference frame
# =============================================================================
def align_target(tgt_cloud, prepared_ref, params, faiss_res=None, logger=None):
    """Returns (T_full 4x4, info dict). T_full maps points in the target's
    native frame into the reference's native frame."""
    def log(msg):
        if logger is not None:
            logger(msg)

    info = {}

    # Cell 9: ICP-voxel downsample + normals BEFORE levelling (matches notebook)
    tgt_icp = downsample_for_icp(tgt_cloud, params)

    # 1. level the target (on the downsampled cloud, as the notebook does)
    T_tgt_level = extract_ground_and_level(
        tgt_icp, params.GROUND_DIST_THRESHOLD,
        params.GROUND_PLANE_ITERS, label="TGT", params=params)
    tgt_levelled = apply_transform(tgt_icp, T_tgt_level)
    tgt_levelled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.NORMAL_RADIUS, max_nn=30))

    # 2. ground-remove + normalize + FPFH
    tgt_above = remove_ground_points(tgt_levelled, params.GROUND_REMOVE_Z)
    tgt_norm, tgt_xy_ctr = normalize_xy(tgt_above)
    tgt_down, tgt_fpfh = preprocess_for_ransac(
        tgt_norm, params.RANSAC_VOXEL,
        params.NORMAL_RADIUS_R, params.FEATURE_RADIUS_R)
    log("FPFH: ref=%d tgt=%d pts" %
        (len(prepared_ref.ref_down.points), len(tgt_down.points)))

    # 3. FAISS correspondences  (src=target, tgt=reference)
    src_corr, tgt_corr = get_correspondences_faiss(
        tgt_fpfh, prepared_ref.ref_fpfh, tgt_down, prepared_ref.ref_down,
        faiss_res=faiss_res, mutual=True)
    n_corr = src_corr.shape[1]
    info["n_correspondences"] = int(n_corr)
    log("FAISS mutual correspondences: %d" % n_corr)
    if n_corr < 3:
        raise RuntimeError(
            "Only %d correspondences (<3). Lower RANSAC_VOXEL." % n_corr)

    # 4. TEASER++ global
    R_te, t_te = run_teaser(
        src_corr, tgt_corr, params.TEASER_NOISE_BOUND,
        params.TEASER_GNC_FACTOR, params.TEASER_MAX_ITER)
    r = TEASERResult(R_te, t_te, src_corr, tgt_corr, params.TEASER_NOISE_BOUND)
    info["teaser_fitness"] = r.fitness
    info["teaser_rmse"]    = r.inlier_rmse
    log("TEASER fitness=%.4f rmse=%.4f" % (r.fitness, r.inlier_rmse))

    # compose global transform in levelled frame
    T_tgt_to_norm = make_T(-tgt_xy_ctr[0], -tgt_xy_ctr[1])
    T_norm_to_ref = make_T(prepared_ref.ref_xy_ctr[0], prepared_ref.ref_xy_ctr[1])
    T_global_levelled = T_norm_to_ref @ r.transformation @ T_tgt_to_norm

    # 5. GICP refinement in levelled frame, init from TEASER
    #    align target_levelled onto ref_levelled
    r1 = icp_pass(tgt_levelled, prepared_ref.ref_levelled, T_global_levelled,
                  params.ICP_DIST_COARSE, params.ICP_MAX_ITER)
    r2 = icp_pass(tgt_levelled, prepared_ref.ref_levelled, r1.transformation,
                  params.ICP_DIST_FINE, params.ICP_MAX_ITER)
    info["icp_fitness"] = r2.fitness
    info["icp_rmse"]    = r2.inlier_rmse
    log("GICP fitness=%.4f rmse=%.4fcm" % (r2.fitness, r2.inlier_rmse * 100))

    # 6. compose back out of levelled frames into native frames:
    #    target_native --T_tgt_level--> target_levelled --r2--> ref_levelled
    #    --inv(T_ref_level)--> ref_native
    T_full = np.linalg.inv(prepared_ref.T_ref_level) @ r2.transformation @ T_tgt_level
    return T_full, info