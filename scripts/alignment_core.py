#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alignment_core.py
"""

import numpy as np
import open3d as o3d
import copy

import teaserpp_python
import faiss
import small_gicp


class AlignParams(object):
    # ---- Voxel / feature scales (FIXED -- these are the values that worked) ----
    ICP_VOXEL        = 0.10      # cloud used for levelling + GICP
    NORMAL_RADIUS    = 0.20      # = ICP_VOXEL * 2
    RANSAC_VOXEL     = 0.30      # cloud used for FPFH
    NORMAL_RADIUS_R  = 0.60      # = RANSAC_VOXEL * 2
    FEATURE_RADIUS_R = 1.50      # = RANSAC_VOXEL * 5

    # ---- TEASER ----
    TEASER_NOISE_BOUND = 0.60    # = RANSAC_VOXEL * 2
    TEASER_GNC_FACTOR  = 1.4
    TEASER_MAX_ITER    = 100

    # ---- GICP (coarse then fine) ----
    ICP_DIST_COARSE = 0.5
    ICP_DIST_FINE   = 0.1
    ICP_MAX_ITER    = 50

    GROUND_REMOVE_Z = 0.30       

    # ---- Ground-plane levelling ----
    # FLOOR_MAX_NORMAL_TILT_DEG = 30
    # MIN_FLOOR_POINTS = 30            
    # FLOOR_SEARCH_PCT   = 40.0   
    # FLOOR_SLAB_M       = 0.10  
    # FLOOR_MIN_SLAB_FRAC = 0.40   
    # FLOOR_BAND_M       = 0.50 
    # FLOOR_FIT_ITERS    = 5      
    # FLOOR_FIT_SIGMA    = 2.5     

    # ---- Diagnostic score only (NOT used for any decision) ----
    SCORE_INLIER_DIST = 0.30     # m; a target point within this of the reference
                                 # counts as covered

    # EDIT IN PARAMS.YAML
    GLOBAL_VOXEL     = 0.10                 # sampling scale for the full-map solve
    GLOBAL_NORMAL_R  = GLOBAL_VOXEL * 3.0   # normals see a stable neighborhood
    GLOBAL_FEATURE_R = GLOBAL_VOXEL * 5.0   # FPFH patch = a meaningful chunk of scene
    GLOBAL_TEASER_NB = GLOBAL_VOXEL * 2.0   # inlier threshold ~ post-downsample scatter


def make_T(tx=0.0, ty=0.0, tz=0.0):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
    return T


def _yaw_tilt(R):
    """(yaw_deg, tilt_deg): yaw about +Z, and how far the rotated +Z axis
    leans from world +Z. tilt ~0 means the two clouds' gravity frames agree;
    a non-trivial tilt is a real DOF that a planar constraint would destroy,
    which is why the global path (align_target) solves full 6-DOF instead."""
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    zc = float(R[2, 2]) / (np.linalg.norm(R[:, 2]) + 1e-12)
    tilt = float(np.degrees(np.arccos(np.clip(zc, -1.0, 1.0))))
    return yaw, tilt


def _tilt_and_z(T):
    """(tilt_deg, tz): how far the transform's local +Z axis leans from world +Z,
    and its Z translation. Used to report what the planar constraint removes."""
    zaxis = np.asarray(T[:3, 2], dtype=float)
    nz = zaxis[2] / (np.linalg.norm(zaxis) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))), float(T[2, 3])


def constrain_planar(T):
    """Project a 4x4 transform onto PLANAR motion: yaw about +Z and XY
    translation only. (any z tilt introduced by TEASER or gicp is error 
    and must be cancelled out)."""
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[0, 0], out[0, 1] = cyaw, -syaw
    out[1, 0], out[1, 1] = syaw, cyaw
    out[0, 3] = T[0, 3]     # keep X translation
    out[1, 3] = T[1, 3]     # keep Y translation
    out[2, 3] = 0.0         # drop Z translation (floors already coincide at Z=0)
    return out


# LEVELLING
def _floor_seed(z, params):
    """Floor height = the lowest significant horizontal slab. Scanning from the
    bottom and taking the first slab that reaches FLOOR_MIN_SLAB_FRAC."""
    z = np.asarray(z)
    if len(z) == 0:
        return 0.0
    lo = float(np.percentile(z, 0.1))
    hi = float(np.percentile(z, params.FLOOR_SEARCH_PCT))
    slab = float(params.FLOOR_SLAB_M)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float(np.median(z))
    edges = np.arange(lo, hi, slab * 0.5)
    if len(edges) == 0:
        return float(np.median(z))
    counts = np.array([np.count_nonzero((z >= e) & (z < e + slab))
                       for e in edges])
    cmax = int(counts.max())
    if cmax <= 0:
        return float(np.median(z))
    thresh = float(params.FLOOR_MIN_SLAB_FRAC) * cmax
    idx = int(np.argmax(counts >= thresh))   # lowest slab meeting the threshold
    return float(edges[idx] + slab * 0.5)


def fit_floor_plane(pts, normals, params):
    """Deterministic floor plane z = a*x + b*y + c.
    Returns (a, b, c, floor_pts, n_floor, horizontal_used)."""
    pts = np.asarray(pts)
    z = pts[:, 2]
    if len(z) < 10:
        h = float(z.min()) if len(z) else 0.0
        return 0.0, 0.0, h, pts, len(pts), False

    # 1. keep only floor-like (near-vertical normal) points -> drop walls
    horizontal_used = False
    cand = pts
    if normals is not None and len(normals) == len(pts):
        nz = np.abs(np.asarray(normals)[:, 2])
        horiz = nz > np.cos(np.radians(params.FLOOR_MAX_NORMAL_TILT_DEG))
        if int(horiz.sum()) >= int(params.MIN_FLOOR_POINTS):
            cand = pts[horiz]
            horizontal_used = True

    # 2. seed the floor height as the lowest significant horizontal band
    seed = _floor_seed(cand[:, 2], params)
    band = float(params.FLOOR_BAND_M)
    P = cand[np.abs(cand[:, 2] - seed) <= band]
    if len(P) < 10:
        return 0.0, 0.0, seed, P, len(P), horizontal_used

    # 3. least-squares plane fit over the floor points, sigma-clipped
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    b = P[:, 2].astype(np.float64).copy()
    coef = np.array([0.0, 0.0, float(seed)])
    for _ in range(max(1, int(params.FLOOR_FIT_ITERS))):
        sol, _r, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
        coef = sol
        resid = b - A @ coef
        s = float(np.std(resid))
        if s < 1e-9:
            break
        keep = np.abs(resid) < params.FLOOR_FIT_SIGMA * s
        if keep.all() or int(keep.sum()) < 10:
            break
        A, b = A[keep], b[keep]

    a, bb, c = float(coef[0]), float(coef[1]), float(coef[2])
    floor_pts = np.c_[A[:, 0], A[:, 1], b]   # final inlier floor points
    return a, bb, c, floor_pts, len(floor_pts), horizontal_used


def level_cloud(pcd, params):
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    a, bb, c, floor_pts, n_floor, horiz_used = fit_floor_plane(pts, normals, params)

    # floor height = fitted plane sampled at the floor centroid (one
    # representative height). No rotation, so no need to transform points first.
    if len(floor_pts) > 0:
        cx = float(np.mean(floor_pts[:, 0]))
        cy = float(np.mean(floor_pts[:, 1]))
        ground_z = float(a * cx + bb * cy + c)
    else:
        ground_z = float(c)

    # diagnostic only: the residual tilt we deliberately do NOT correct
    normal = np.array([a, bb, -1.0])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal               # point up
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))

    T = np.eye(4)
    T[2, 3] = -ground_z                # Z translation ONLY

    out = copy.deepcopy(pcd)
    out.transform(T)
    meta = {"tilt_deg": tilt_deg, "n_floor": int(n_floor),
            "horizontal_used": bool(horiz_used), "ground_z": ground_z,
            "normal": [float(normal[0]), float(normal[1]), float(normal[2])]}
    return out, T, meta


# =============================================================================
# Features / correspondences / solvers 
# =============================================================================
def _remove_ground(pcd, z_thresh):
    pts = np.asarray(pcd.points)
    keep = pts[:, 2] > z_thresh
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[keep])
    return out


def normalize_xy(pcd):
    """Subtract the XY bounding-box centre so TEASER sees both clouds centred at
    the origin (as in the notebook). Returns (shifted_pcd, (cx, cy))."""
    ctr = pcd.get_axis_aligned_bounding_box().get_center()
    T = make_T(-ctr[0], -ctr[1], 0.0)
    out = copy.deepcopy(pcd)
    out.transform(T)
    return out, (float(ctr[0]), float(ctr[1]))


def _prep_icp_cloud(cloud, voxel, normal_r):
    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30))
    return down


def _score_alignment(ref_lev, src_pts, T_lev, params):
    """Diagnostic coverage/RMSE score against the levelled reference (reported only; drives no decision)."""
    kd = o3d.geometry.KDTreeFlann(ref_lev)
    rs = np.random.RandomState(0)
    samp = src_pts if len(src_pts) <= 3000 else \
        src_pts[rs.choice(len(src_pts), 3000, replace=False)]
    q = (T_lev[:3, :3] @ samp.T).T + T_lev[:3, 3]
    d = np.array([np.sqrt(kd.search_knn_vector_3d(x, 1)[2][0]) for x in q])
    inl = d < params.SCORE_INLIER_DIST
    coverage = float(inl.mean())
    rmse = float(np.sqrt((d[inl] ** 2).mean())) if inl.any() else float("inf")
    return coverage, rmse, d


def _score_global(src_pts, ref_pts, T, inl_dist):
    """(coverage, inlier_rmse): fraction of (subsampled) src landing within
    inl_dist of ref after applying T, and the RMSE over those inliers.
    Diagnostic only -- drives no decision inside align_target."""
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)
    kd = o3d.geometry.KDTreeFlann(ref_pcd)
    rs = np.random.RandomState(0)
    samp = src_pts if len(src_pts) <= 3000 else \
        src_pts[rs.choice(len(src_pts), 3000, replace=False)]
    q = (T[:3, :3] @ samp.T).T + T[:3, 3]
    d = np.array([np.sqrt(kd.search_knn_vector_3d(x, 1)[2][0]) for x in q])
    inl = d < inl_dist
    cov = float(inl.mean())
    rmse = float(np.sqrt((d[inl] ** 2).mean())) if inl.any() else float("inf")
    return cov, rmse


def _fpfh(pcd, voxel, normal_r, feat_r):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=feat_r, max_nn=100))
    return down, f


def _correspondences(src_pcd, src_f, ref_pcd, ref_f, faiss_res=None):
    """Mutual nearest-neighbour FPFH matching -> (3,N) src, (3,N) ref."""
    A = np.ascontiguousarray(np.array(src_f.data).T, dtype=np.float32)
    B = np.ascontiguousarray(np.array(ref_f.data).T, dtype=np.float32)
    sp = np.asarray(src_pcd.points)
    rp = np.asarray(ref_pcd.points)

    d = A.shape[1]
    try:
        use_gpu = (hasattr(faiss, "StandardGpuResources")
                   and faiss.get_num_gpus() > 0)
    except Exception:
        use_gpu = False
    if use_gpu:
        if faiss_res is None:
            faiss_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(faiss_res, 0, faiss.IndexFlatL2(d))
    else:
        index = faiss.IndexFlatL2(d)

    index.add(B)
    _, fwd = index.search(A, 1)
    fwd = fwd.flatten()
    index.reset()
    index.add(A)
    _, bwd = index.search(B[fwd], 1)
    bwd = bwd.flatten()
    mutual = (bwd == np.arange(len(fwd)))
    return sp[mutual].T, rp[fwd[mutual]].T


def _teaser(src_corr, ref_corr, noise_bound, params):
    p = teaserpp_python.RobustRegistrationSolver.Params()
    p.cbar2 = 1.0
    p.noise_bound = noise_bound
    p.estimate_scaling = False
    p.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver
        .ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
    p.rotation_gnc_factor = params.TEASER_GNC_FACTOR
    p.rotation_max_iterations = params.TEASER_MAX_ITER
    p.rotation_cost_threshold = 1e-12
    s = teaserpp_python.RobustRegistrationSolver(p)
    s.solve(src_corr, ref_corr)
    sol = s.getSolution()
    T = np.eye(4)
    T[:3, :3] = sol.rotation
    T[:3, 3] = sol.translation
    return T


def _gicp(src_pts, ref_pts, init_T, dist, params):
    """GICP refine on the levelled clouds

    align(target=ref, source=tgt) returns
    T_target_source; we invert it to get the tgt->ref transform. `init_T` maps
    tgt->ref, so its inverse is passed as init_T_target_source."""
    src_sg = small_gicp.PointCloud(
        np.ascontiguousarray(src_pts, dtype=np.float64))    # target being aligned
    tgt_sg = small_gicp.PointCloud(
        np.ascontiguousarray(ref_pts, dtype=np.float64))    # reference
    r = small_gicp.align(
        tgt_sg, src_sg,
        init_T_target_source=np.linalg.inv(init_T),
        registration_type="GICP",
        max_correspondence_distance=dist,
        num_threads=8,
        max_iterations=params.ICP_MAX_ITER)
    return np.linalg.inv(r.T_target_source)


class PreparedReference(object):
    def __init__(self, ref_cloud, params):
        self.params = params
        self.ref_cloud = ref_cloud
        self.ref_raw = np.asarray(ref_cloud.points)


# =============================================================================
# full alignment logic
# =============================================================================
def align_target(tgt_cloud, prepared_ref, params, faiss_res=None, logger=None,
                  init_T=None, skip_teaser=False):
    """
        returns 4x4 tgt -> ref
        voxel downsample -> normals -> FPFH -> mutual-NN correspondences
          -> TEASER++ (full 6-DOF)  -> GICP refine (coarse -> fine)
    """
    def log(m):
        if logger:
            logger(m)

    p = params
    v = p.GLOBAL_VOXEL
    ref_cloud = prepared_ref.ref_cloud

    if skip_teaser and init_T is not None:
        # ---- Seam B fast path: seed GICP directly, skip feature matching ----
        n_corr = n_inliers = 0
        teaser_yaw = None
        T = np.array(init_T, dtype=np.float64)
        yaw, tilt = _yaw_tilt(T[:3, :3])
        log("Seam-B seed: yaw=%.2f tilt=%.2f (skipped FPFH/TEASER)" % (yaw, tilt))
        src_d = tgt_cloud.voxel_down_sample(v)
        ref_d = ref_cloud.voxel_down_sample(v)
    else:
        # ---- 1. FPFH on the full clouds (NO level_cloud / NO _remove_ground) ----
        src_d, src_f = _fpfh(tgt_cloud, v, p.GLOBAL_NORMAL_R, p.GLOBAL_FEATURE_R)
        ref_d, ref_f = _fpfh(ref_cloud, v, p.GLOBAL_NORMAL_R, p.GLOBAL_FEATURE_R)
        log("FPFH: tgt=%d ref=%d pts (voxel=%.2f)"
            % (len(src_d.points), len(ref_d.points), v))

        # ---- 2. mutual-NN FPFH correspondences (tgt -> ref direction) ----
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
        teaser_yaw = yaw
        log("TEASER yaw=%.2f tilt=%.2f inliers=%d/%d (%.1f%%) t=[%.2f %.2f %.2f]"
            % (yaw, tilt, n_inliers, n_corr, 100.0 * n_inliers / max(n_corr, 1),
               T[0, 3], T[1, 3], T[2, 3]))

    # ---- 4. GICP refine, coarse -> fine (NO planar constraint) ----
    src_pts = np.ascontiguousarray(np.asarray(src_d.points), dtype=np.float64)
    ref_pts = np.ascontiguousarray(np.asarray(ref_d.points), dtype=np.float64)
    for dist in (v * 5.0, v * 2.0, v * 1.0):
        T = _gicp(src_pts, ref_pts, T, dist, p)
    yaw, tilt = _yaw_tilt(T[:3, :3])

    # ---- 5. coverage diagnostic + return ----
    coverage, rmse = _score_global(src_pts, ref_pts, T, v * 2.0)
    log("post-GICP yaw=%.2f tilt=%.2f cov=%.2f rmse=%.2f t=[%.2f %.2f %.2f]"
        % (yaw, tilt, coverage, rmse, T[0, 3], T[1, 3], T[2, 3]))

    info = {
        "path": "seeded" if (skip_teaser and init_T is not None) else "teaser",
        "n_correspondences": int(n_corr),
        "teaser_inliers": int(n_inliers),
        "teaser_yaw_deg": teaser_yaw,
        "yaw_deg": yaw,
        "tilt_deg": tilt,
        "coverage": coverage,
        "inlier_rmse": rmse,
        "tx": float(T[0, 3]), "ty": float(T[1, 3]), "tz": float(T[2, 3]),
    }
    return T, info