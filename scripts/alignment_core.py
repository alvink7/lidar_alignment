#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alignment_core.py
-----------------
Core point-cloud alignment pipeline, extracted verbatim (in behaviour) from the
multibag_mapping notebook so the live ROS node and the offline reference-prep
script share ONE implementation.

Pipeline per alignment:
    level (gravity vector OR ground plane -> Z up, ground at 0)
      -> remove ground points
      -> normalize XY to a SHARED origin (the reference center)
      -> FPFH features
      -> FAISS GPU mutual-NN correspondences
      -> TEASER++ global registration
      -> small_gicp GICP refinement (coarse then fine)
    => 4x4 transform mapping TARGET frame into REFERENCE frame.

This module is pure Python / numpy / open3d / faiss / teaserpp_python /
small_gicp.  It imports NOTHING from rospy, so it can be unit-tested off-robot.

LEVELLING: prefer FAST-LIO's gravity vector (deterministic, no RANSAC). Pass a
gravity vector (gravity direction in the cloud frame, pointing DOWN) into
PreparedReference / align_target and levelling uses it directly. If no gravity
is supplied, falls back to the horizontal-constrained plane fit.
"""

import numpy as np
import open3d as o3d
import copy

import teaserpp_python
import faiss
import small_gicp

class AlignParams(object):
    ICP_VOXEL    = 0.10
    RANSAC_VOXEL = 0.20
    VOXEL_SIZE   = 0.05

    GROUND_DIST_THRESHOLD = 0.15
    GROUND_REMOVE_Z       = 0.30

    NORMAL_RADIUS_R  = RANSAC_VOXEL * 2
    FEATURE_RADIUS_R = 1.5

    TEASER_NOISE_BOUND = RANSAC_VOXEL * 2.0
    TEASER_GNC_FACTOR  = 1.4
    TEASER_MAX_ITER    = 100

    ICP_DIST_COARSE = 0.5
    ICP_DIST_FINE   = 0.1
    ICP_MAX_ITER    = 50
    NORMAL_RADIUS   = ICP_VOXEL * 2

    GROUND_PLANE_ITERS = 200
    GROUND_MAX_TILT_DEG = 30.0
    GROUND_PLANE_ATTEMPTS = 12
    GICP_DOWNSAMPLE_RES = 0.05
    CONSTRAIN_TO_YAW = True


    LEVEL_MODE = "gravity_height"
    FLOOR_Z_PERCENTILE = 2.0
    
    USE_GRAVITY_LEVELLING = True

    XY_NORMALIZE_MODE = "reference"


# =============================================================================
# Levelling helpers
# =============================================================================
def _R_align_normal_to_z(normal):
    """Rodrigues rotation mapping `normal` -> +Z. Deterministic, no sampling.
    Shared by both the gravity path and the plane-fit path so there is exactly
    one tested rotation builder."""
    n = np.asarray(normal, dtype=np.float64).copy()
    nrm = np.linalg.norm(n)
    if nrm < 1e-12:
        return np.eye(3)
    n /= nrm
    if n[2] < 0:
        n = -n
    z_axis = np.array([0.0, 0.0, 1.0])
    axis   = np.cross(n, z_axis)
    angle  = np.arccos(np.clip(np.dot(n, z_axis), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    axis /= np.linalg.norm(axis)
    K = np.array([
        [ 0,       -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _level_from_normal(pcd, normal, floor_percentile=2.0):
    """Build a 4x4 T_level that rotates `normal` upright to +Z and puts the
    ground at Z=0 (low-percentile height in the rotated frame). The height comes
    from the lowest-Z points, which is the deterministic floor-height estimate."""
    R = _R_align_normal_to_z(normal)
    T_level = np.eye(4)
    T_level[:3, :3] = R
    pts_rot_z = (np.asarray(pcd.points) @ R.T)[:, 2]
    ground_z = np.percentile(pts_rot_z, floor_percentile)
    T_level[2, 3] = -ground_z
    return T_level


def level_from_gravity(pcd, gravity, params=None):
    """Level using a gravity vector instead of a plane fit."""
    pct = getattr(params, "FLOOR_Z_PERCENTILE", 2.0) if params else 2.0
    return _level_from_normal(
        pcd, -np.asarray(gravity, dtype=np.float64), floor_percentile=pct)


def level_gravity_height(pcd, gravity=None, params=None):
    """PRIMARY levelling for gravity-aligned (FAST-LIO) clouds."""
    pct = getattr(params, "FLOOR_Z_PERCENTILE", 2.0) if params else 2.0
    if gravity is not None:
        up = -np.asarray(gravity, dtype=np.float64)
    else:
        up = np.array([0.0, 0.0, 1.0])     # trust gravity-alignment: no rotation
    return _level_from_normal(pcd, up, floor_percentile=pct)


# =============================================================================
# Ground extraction / levelling 
# =============================================================================
def _find_horizontal_plane(pcd, dist_threshold, plane_iters, params, label=""):
    
    max_tilt = getattr(params, "GROUND_MAX_TILT_DEG", 30.0)
    work = pcd
    remaining_idx = np.arange(len(pcd.points))
    # track only the best HORIZONTAL candidate (by inlier count) as a tie-break;
    # tilted planes are never eligible to be returned.
    best_horiz = None
    for attempt in range(getattr(params, "GROUND_PLANE_ATTEMPTS", 12)):
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
        global_inl = remaining_idx[inl]        # map back to original indices
        if tilt <= max_tilt:
            # horizontal: this is the floor. Keep the largest such plane.
            if best_horiz is None or len(global_inl) > len(best_horiz[1]):
                best_horiz = (n, global_inl)
            return best_horiz                  # first horizontal hit is enough
       
        mask = np.ones(len(work.points), dtype=bool)
        mask[inl] = False
        remaining_idx = remaining_idx[mask]
        work = work.select_by_index(np.where(mask)[0])
    # No horizontal plane found within the attempt budget.
    if best_horiz is not None:
        return best_horiz
    return np.array([0.0, 0.0, 1.0]), np.arange(len(pcd.points))


def extract_ground_and_level(pcd, dist_threshold, plane_iters, label="",
                            params=None, gravity=None):
    """Return 4x4 T_level that rotates cloud upright and puts the ground at Z=0.
    Behaviour is controlled by params.LEVEL_MODE:

    - "gravity_height": trust the fast lio gravity-alignment. """
    if params is None:
        params = AlignParams()

    mode = getattr(params, "LEVEL_MODE", "gravity_height")

    if mode == "gravity_height":
        return level_gravity_height(pcd, gravity=gravity, params=params)

    # ---- opt-in: RANSAC plane fit ----
    normal, inliers = _find_horizontal_plane(
        pcd, dist_threshold, plane_iters, params, label)
    pct = getattr(params, "FLOOR_Z_PERCENTILE", 2.0)
    return _level_from_normal(pcd, normal, floor_percentile=pct)


def apply_transform(pcd, T):
    out = copy.deepcopy(pcd)
    out.transform(T)
    return out


# =============================================================================
# Ground removal / XY normalize / FPFH 
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


def normalize_xy(pcd, center=None):
    """Recenter a cloud in XY by subtracting `center` (a length-2 or length-3
    array). If `center` is None, uses this cloud's own bbox center (only correct when a single cloud is centered in isolation).

    Returns (recentered_pcd, center_used).."""
    if center is None:
        center = pcd.get_axis_aligned_bounding_box().get_center()
    center = np.asarray(center, dtype=np.float64)
    T = np.eye(4)
    T[0, 3] = -center[0]
    T[1, 3] = -center[1]
    return apply_transform(pcd, T), center


def _resolve_feature_radius(params):
    fr = getattr(params, "FEATURE_RADIUS_R", None)
    if fr is None:
        fr = params.RANSAC_VOXEL * 5
    return fr


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


def constrain_to_yaw(T):
    """Project a transform (in the LEVELLED frame, where both clouds have their
    floor flat) onto pure yaw + translation. Levelling already fixes pitch/roll,
    so any pitch/roll in a global-registration result is spurious and a source
    of wrong, non-deterministic solutions on sparse data. This zeroes it out."""
    R = T[:3, :3].copy()
    # yaw = rotation about Z. Extract it from the forward (x) axis projected to XY.
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    out = np.eye(4)
    out[:3, :3] = Rz
    out[:3, 3] = T[:3, 3]     # keep translation (incl. any Z offset)
    return out


# =============================================================================
# FAISS GPU correspondences 
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
    """Mutual NN FPFH matching. Returns (src_corr, tgt_corr) as (3,N)."""
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
# TEASER++
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
# small_gicp ICP
# =============================================================================
class _ICPResult(object):
    def __init__(self, T, fitness, rmse):
        self.transformation = T
        self.fitness        = fitness
        self.inlier_rmse    = rmse


def icp_pass(src, tgt, initial_transform, dist_threshold, max_iter):
    src_pts = np.ascontiguousarray(np.asarray(src.points), dtype=np.float64)
    tgt_pts = np.ascontiguousarray(np.asarray(tgt.points), dtype=np.float64)

    # small_gicp.align (raw-points overload) does its OWN preprocessing:
    # downsampling + normal/covariance estimation. Its default
    # downsampling_resolution=0.25 re-thins already-sparse clouds, which starves
    # GICP of the covariances it needs (=> nan error, no refinement). Our clouds
    # are already ICP_VOXEL-downsampled upstream, so set a small resolution to
    # avoid destructive re-downsampling while still letting it build covariances.
    ds = getattr(AlignParams, "GICP_DOWNSAMPLE_RES", 0.0)
    res = small_gicp.align(
        tgt_pts, src_pts,
        init_T_target_source=np.linalg.inv(initial_transform),
        registration_type="GICP",
        downsampling_resolution=ds,
        max_correspondence_distance=dist_threshold,
        num_threads=8,
        max_iterations=max_iter,
    )
    T = np.linalg.inv(res.T_target_source)
    n_in = res.num_inliers
    fitness = n_in / max(len(src_pts), 1)
    if n_in > 0 and np.isfinite(res.error):
        rmse = float(np.sqrt(res.error / n_in))
    else:
        rmse = float("inf")
    return _ICPResult(T, fitness, rmse)


# =============================================================================
# ICP-voxel downsample + normals 
# =============================================================================
def downsample_for_icp(pcd, params):
    down = pcd.voxel_down_sample(params.ICP_VOXEL)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.NORMAL_RADIUS, max_nn=30))
    return down

# Reference preprocessing
class PreparedReference(object):
    """Everything about the reference that the per-target alignment needs,
    computed once.

    Pass `gravity` (gravity direction in the reference cloud's frame, pointing
    DOWN -- from FAST-LIO) to level the reference deterministically. The
    reference and each target have different camera_init origins, so each MUST
    be levelled with its OWN gravity vector; the reference's is captured here."""
    def __init__(self, ref_cloud, params, gravity=None):
        self.params = params
        self.gravity = None if gravity is None else np.asarray(
            gravity, dtype=np.float64)
        # Cell 9: ICP-voxel downsample + normals BEFORE levelling
        ref_icp = downsample_for_icp(ref_cloud, params)
        # Cell 10: level the downsampled cloud (gravity if available, else fit)
        self.T_ref_level = extract_ground_and_level(
            ref_icp, params.GROUND_DIST_THRESHOLD,
            params.GROUND_PLANE_ITERS, label="REF", params=params,
            gravity=self.gravity)
        self.ref_levelled = apply_transform(ref_icp, self.T_ref_level)
        self.ref_levelled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.NORMAL_RADIUS, max_nn=30))
        # ground-removed / normalized / FPFH for global registration
        ref_above = remove_ground_points(self.ref_levelled, params.GROUND_REMOVE_Z)

        # XY normalization: record the reference center so the TARGET can be
        # recentered by the SAME value (see XY_NORMALIZE_MODE). This is the fix
        # for the bbox-center bug: never let the two clouds use different centers.
        xy_mode = getattr(params, "XY_NORMALIZE_MODE", "reference")
        if xy_mode == "none":
            self.ref_xy_ctr = np.zeros(3)
            ref_norm = ref_above
        else:
            ref_norm, self.ref_xy_ctr = normalize_xy(ref_above, center=None)

        feat_r = _resolve_feature_radius(params)
        self.ref_down, self.ref_fpfh = preprocess_for_ransac(
            ref_norm, params.RANSAC_VOXEL,
            params.NORMAL_RADIUS_R, feat_r)


# Full target alignment  ->  T mapping target frame into reference frame
def align_target(tgt_cloud, prepared_ref, params, faiss_res=None, logger=None,
                 gravity=None):
    """Returns (T_full 4x4, info dict). T_full maps points in the target's
    native frame into the reference's native frame.

    Pass `gravity` (gravity direction in the TARGET cloud's own frame, pointing
    DOWN -- from FAST-LIO, ideally sampled at the END of the accumulation window
    when the estimate has converged). This levels the target deterministically,
    removing the RANSAC plane fit and its run-to-run variation. If omitted, the
    plane-fit fallback is used."""
    def log(msg):
        if logger is not None:
            logger(msg)

    info = {}
    grav = None if gravity is None else np.asarray(gravity, dtype=np.float64)
    info["used_gravity_levelling"] = bool(
        grav is not None and getattr(params, "USE_GRAVITY_LEVELLING", True))

    # ICP-voxel downsample + normals
    tgt_icp = downsample_for_icp(tgt_cloud, params)

    # level the target (gravity if available, else plane fit)
    T_tgt_level = extract_ground_and_level(
        tgt_icp, params.GROUND_DIST_THRESHOLD,
        params.GROUND_PLANE_ITERS, label="TGT", params=params, gravity=grav)
    tgt_levelled = apply_transform(tgt_icp, T_tgt_level)
    tgt_levelled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.NORMAL_RADIUS, max_nn=30))

    # ground-remove + normalize + FPFH
    tgt_above = remove_ground_points(tgt_levelled, params.GROUND_REMOVE_Z)

    # XY normalization: recenter the target by the SAME center used
    # for the reference (or skip entirely), NOT by the target's own bbox center.
    # Using each cloud's own bbox center injects a spurious relative offset when
    # the clouds cover different regions (partial overlap), which fights TEASER.
    xy_mode = getattr(params, "XY_NORMALIZE_MODE", "reference")
    if xy_mode == "none":
        tgt_norm = tgt_above
        tgt_xy_ctr = np.zeros(3)
        shared_ctr = np.zeros(3)
    elif xy_mode == "self":
        tgt_norm, tgt_xy_ctr = normalize_xy(tgt_above, center=None)
        shared_ctr = prepared_ref.ref_xy_ctr
    else:  # "reference" (default): both clouds recentered by the reference center
        shared_ctr = prepared_ref.ref_xy_ctr
        tgt_norm, tgt_xy_ctr = normalize_xy(tgt_above, center=shared_ctr)

    feat_r = _resolve_feature_radius(params)
    tgt_down, tgt_fpfh = preprocess_for_ransac(
        tgt_norm, params.RANSAC_VOXEL,
        params.NORMAL_RADIUS_R, feat_r)
    log("FPFH: ref=%d tgt=%d pts (xy_mode=%s, feat_r=%.2f)" %
        (len(prepared_ref.ref_down.points), len(tgt_down.points),
         xy_mode, feat_r))

    # faiss correspondences  (src=target, tgt=reference)
    src_corr, tgt_corr = get_correspondences_faiss(
        tgt_fpfh, prepared_ref.ref_fpfh, tgt_down, prepared_ref.ref_down,
        faiss_res=faiss_res, mutual=True)
    n_corr = src_corr.shape[1]
    info["n_correspondences"] = int(n_corr)
    log("FAISS mutual correspondences: %d" % n_corr)
    if n_corr < 3:
        raise RuntimeError(
            "Only %d correspondences (<3). Lower RANSAC_VOXEL." % n_corr)

    # TEASER++ global
    R_te, t_te = run_teaser(
        src_corr, tgt_corr, params.TEASER_NOISE_BOUND,
        params.TEASER_GNC_FACTOR, params.TEASER_MAX_ITER)
    r = TEASERResult(R_te, t_te, src_corr, tgt_corr, params.TEASER_NOISE_BOUND)
    info["teaser_fitness"] = r.fitness
    info["teaser_rmse"]    = r.inlier_rmse
    log("TEASER fitness=%.4f rmse=%.4f" % (r.fitness, r.inlier_rmse))

    # compose global transform in levelled frame.
    # In "reference"/"self" modes both clouds were shifted by shared_ctr; in
    # "none" mode shared_ctr is zero so these are identities.
    T_tgt_to_norm = make_T(-shared_ctr[0], -shared_ctr[1])
    T_norm_to_ref = make_T(shared_ctr[0], shared_ctr[1])
    T_global_levelled = T_norm_to_ref @ r.transformation @ T_tgt_to_norm

    # zero out teaser inaccuracies
    if getattr(params, "CONSTRAIN_TO_YAW", True):
        T_global_levelled = constrain_to_yaw(T_global_levelled)

    # GICP refinement in levelled frame
    r1 = icp_pass(tgt_levelled, prepared_ref.ref_levelled, T_global_levelled,
                  params.ICP_DIST_COARSE, params.ICP_MAX_ITER)
    r2 = icp_pass(tgt_levelled, prepared_ref.ref_levelled, r1.transformation,
                  params.ICP_DIST_FINE, params.ICP_MAX_ITER)
    info["icp_fitness"] = r2.fitness
    info["icp_rmse"]    = r2.inlier_rmse
    log("GICP fitness=%.4f rmse=%.4fcm" % (r2.fitness, r2.inlier_rmse * 100))

    # compose back out of levelled frames into native frames:
    #    target_native --T_tgt_level--> target_levelled --r2--> ref_levelled
    #    --inv(T_ref_level)--> ref_native
    #    r2 is in the levelled frame, so constrain it to yaw too -- GICP can
    #    re-introduce small spurious pitch/roll during refinement.
    r2_levelled = r2.transformation
    if getattr(params, "CONSTRAIN_TO_YAW", True):
        r2_levelled = constrain_to_yaw(r2_levelled)
    T_full = np.linalg.inv(prepared_ref.T_ref_level) @ r2_levelled @ T_tgt_level
    return T_full, info