#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def extract_ground_and_level(pcd, dist_threshold, plane_iters, label=""):
    """Fit ground plane; return 4x4 T_level that rotates cloud upright and puts
    the ground at Z=0."""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_threshold,
        ransac_n=3,
        num_iterations=plane_iters,
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

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


def get_correspondences_faiss(src_fpfh, tgt_fpfh, src_pcd, tgt_pcd,
                              faiss_res=None, mutual=True):
    """Mutual NN FPFH matching on GPU. Returns (src_corr, tgt_corr) as (3,N).
    Pass a persistent faiss.StandardGpuResources() in faiss_res to avoid
    re-allocating GPU scratch each call."""
    src_feat = np.ascontiguousarray(np.array(src_fpfh.data).T, dtype=np.float32)
    tgt_feat = np.ascontiguousarray(np.array(tgt_fpfh.data).T, dtype=np.float32)
    src_pts  = np.asarray(src_pcd.points)
    tgt_pts  = np.asarray(tgt_pcd.points)

    d = src_feat.shape[1]
    if faiss_res is None:
        faiss_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, faiss.IndexFlatL2(d))

    gpu_index.add(tgt_feat)
    _, fwd = gpu_index.search(src_feat, 1)
    fwd = fwd.flatten()

    if mutual:
        gpu_index.reset()
        gpu_index.add(src_feat)
        _, bwd = gpu_index.search(tgt_feat[fwd], 1)
        bwd = bwd.flatten()
        mask = (bwd == np.arange(len(fwd)))
        return src_pts[mask].T, tgt_pts[fwd[mask]].T

    return src_pts.T, tgt_pts[fwd].T


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



def downsample_for_icp(pcd, params):
    down = pcd.voxel_down_sample(params.ICP_VOXEL)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.NORMAL_RADIUS, max_nn=30))
    return down


# reference preprocessing
class PreparedReference(object):
    """Everything about the reference that the per-target alignment needs,
    computed once."""
    def __init__(self, ref_cloud, params):
        self.params = params

        ref_icp = downsample_for_icp(ref_cloud, params)

        self.T_ref_level = extract_ground_and_level(
            ref_icp, params.GROUND_DIST_THRESHOLD,
            params.GROUND_PLANE_ITERS, label="REF")
        self.ref_levelled = apply_transform(ref_icp, self.T_ref_level)
        self.ref_levelled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.NORMAL_RADIUS, max_nn=30))

        ref_above = remove_ground_points(self.ref_levelled, params.GROUND_REMOVE_Z)
        ref_norm, self.ref_xy_ctr = normalize_xy(ref_above)
        self.ref_down, self.ref_fpfh = preprocess_for_ransac(
            ref_norm, params.RANSAC_VOXEL,
            params.NORMAL_RADIUS_R, params.FEATURE_RADIUS_R)


# full aligment logic
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
        params.GROUND_PLANE_ITERS, label="TGT")
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
