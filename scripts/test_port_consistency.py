#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_port_consistency.py

Verifies the global-registration port is CONSISTENT across:
  (1) global_register.register_global  (standalone)
  (2) alignment_core.align_target       (ported into the core)
  (3) the node's convention chain       (align_target -> matrix_to_transform_stamped)

and that the recovered transform matches ground truth in the CORRECT direction
(target -> reference). A src/ref mix-up would show up as: the two paths
disagreeing, the coverage collapsing when T is applied the intended way, or the
transform matching inv(GT) instead of GT.
"""
import os
import numpy as np
import open3d as o3d

from alignment_core import AlignParams, PreparedReference, align_target
from global_register import register_global
from ros_cloud_utils import matrix_to_transform_stamped  # convention under test

MAPS = os.path.expanduser("~/catkin_ws/maps")
TARGET = os.path.join(MAPS, "test_target.pcd")      # the cloud that gets moved
REF    = os.path.join(MAPS, "reference_jul1.pcd")   # the fixed map to localize INTO
GT_FILE = "gt_run4_T.txt"                            # GT maps target -> reference


def yaw_of(T):
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def coverage(src_pts, ref_pts, T, inl):
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)
    kd = o3d.geometry.KDTreeFlann(ref_pcd)
    rs = np.random.RandomState(0)
    samp = src_pts if len(src_pts) <= 3000 else src_pts[rs.choice(len(src_pts), 3000, replace=False)]
    q = (T[:3, :3] @ samp.T).T + T[:3, 3]
    d = np.array([np.sqrt(kd.search_knn_vector_3d(x, 1)[2][0]) for x in q])
    return float((d < inl).mean())


def load_gt():
    import re
    txt = open(GT_FILE).read()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    return np.array([float(n) for n in nums[:16]]).reshape(4, 4)


def main():
    p = AlignParams()
    print("GLOBAL_VOXEL = %.3f" % p.GLOBAL_VOXEL)
    target = o3d.io.read_point_cloud(TARGET)
    ref = o3d.io.read_point_cloud(REF)
    tgt_pts = np.asarray(target.points)
    ref_pts = np.asarray(ref.points)

    # ---- (1) standalone ----
    T_std, _ = register_global(target, ref, p)

    # ---- (2) core, exactly as the NODE calls it: align_target(target, prepared_ref) ----
    prepared_ref = PreparedReference(ref, p)
    T_core, _ = align_target(target, prepared_ref, p)

    # ---- (3) GT ----
    gt = load_gt()

    print("\n--- transforms (target -> reference) ---")
    print("standalone yaw=%.2f t=%s" % (yaw_of(T_std), np.round(T_std[:3, 3], 3)))
    print("core       yaw=%.2f t=%s" % (yaw_of(T_core), np.round(T_core[:3, 3], 3)))
    print("GT         yaw=%.2f t=%s" % (yaw_of(gt), np.round(gt[:3, 3], 3)))

    ok = True

    # CHECK A: core and standalone must be identical (same logic, same inputs)
    dA = float(np.abs(T_core - T_std).max())
    print("\n[A] core vs standalone   max|dT| = %.2e  %s"
          % (dA, "OK" if dA < 1e-6 else "FAIL"))
    ok &= dA < 1e-6

    # CHECK B: core must match GT (not inv(GT)) -> correct direction baked in
    dyaw = (yaw_of(T_core) - yaw_of(gt) + 180) % 360 - 180
    dxy = float(np.hypot(T_core[0, 3] - gt[0, 3], T_core[1, 3] - gt[1, 3]))
    dyaw_inv = (yaw_of(T_core) - yaw_of(np.linalg.inv(gt)) + 180) % 360 - 180
    print("[B] core vs GT           dyaw=%+.2f deg  dxy=%.2f m   (vs invGT dyaw=%+.2f)"
          % (dyaw, dxy, dyaw_inv))
    matches_gt = abs(dyaw) < 5.0 and dxy < 0.6
    matches_inv = abs(dyaw_inv) < 5.0
    print("    matches GT=%s  matches inv(GT)=%s  -> %s"
          % (matches_gt, matches_inv,
             "OK (correct direction)" if (matches_gt and not matches_inv)
             else "FAIL (WRONG DIRECTION)"))
    ok &= matches_gt and not matches_inv

    # CHECK C: applying T the intended way (target->ref) must give HIGH coverage;
    # the inverse must give LOW coverage. This is the physical direction proof.
    cov_fwd = coverage(tgt_pts, ref_pts, T_core, p.GLOBAL_VOXEL * 2)
    cov_inv = coverage(tgt_pts, ref_pts, np.linalg.inv(T_core), p.GLOBAL_VOXEL * 2)
    print("[C] coverage  T(target->ref)=%.2f   inv=%.2f   %s"
          % (cov_fwd, cov_inv, "OK" if cov_fwd > cov_inv and cov_fwd > 0.4 else "FAIL"))
    ok &= cov_fwd > cov_inv and cov_fwd > 0.4

    # CHECK D: node convention -- matrix_to_transform_stamped(T, ref, target)
    # must encode T unchanged (parent=ref, child=target, maps child->parent).
    ts = matrix_to_transform_stamped(T_core, "reference_map", "target",
                                     stamp=None)
    import tf.transformations as tft
    q = [ts.transform.rotation.x, ts.transform.rotation.y,
         ts.transform.rotation.z, ts.transform.rotation.w]
    T_tf = tft.quaternion_matrix(q)
    T_tf[0, 3] = ts.transform.translation.x
    T_tf[1, 3] = ts.transform.translation.y
    T_tf[2, 3] = ts.transform.translation.z
    dD = float(np.abs(T_tf - T_core).max())
    print("[D] node TF  frame_id=%s child=%s  round-trip max|dT|=%.2e  %s"
          % (ts.header.frame_id, ts.child_frame_id, dD,
             "OK" if (ts.header.frame_id == "reference_map"
                      and ts.child_frame_id == "target" and dD < 1e-9) else "FAIL"))
    ok &= (ts.header.frame_id == "reference_map"
           and ts.child_frame_id == "target" and dD < 1e-9)

    print("\n==== %s ====" % ("ALL CONSISTENT" if ok else "INCONSISTENCY DETECTED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
