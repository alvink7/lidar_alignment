#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_alignment.py
---------------------
Settles "is the alignment actually bad, or do the metrics just look bad?" and
specifically tests the indoor ground-plane failure mode.

1. Fits the ground plane MULTIPLE times per cloud and reports whether the normal
   is stable. Unstable normals (changing between runs) = RANSAC picking
   different planes (floor vs wall) — the classic indoor failure.
2. Runs the full alignment.
3. Reports overlap fitness at several distance thresholds.
4. Opens an Open3D window: reference (blue) vs transformed target (orange).

Usage:
  python3 diagnose_alignment.py REF.pcd TGT.pcd
  python3 diagnose_alignment.py REF.pcd TGT.pcd --plane-trials 5
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
from alignment_core import (AlignParams, PreparedReference, align_target,
                            downsample_for_icp, extract_ground_and_level)


def fit_ground_normal(pcd_down, params):
    pm, inl = pcd_down.segment_plane(
        distance_threshold=params.GROUND_DIST_THRESHOLD,
        ransac_n=3, num_iterations=params.GROUND_PLANE_ITERS)
    n = np.array(pm[:3]); n /= np.linalg.norm(n)
    if n[2] < 0: n = -n
    return n, len(inl), len(pcd_down.points)


def ground_stability(pcd, params, label, trials):
    """Fit the plane `trials` times; report each normal and the spread.
    An unstable normal between trials indicates RANSAC is choosing among
    competing planes (indoor floor-vs-wall ambiguity)."""
    icp = downsample_for_icp(pcd, params)
    print("  [%s] fitting ground plane %d times:" % (label, trials))
    normals = []
    for i in range(trials):
        n, ninl, ntot = fit_ground_normal(icp, params)
        # angle from vertical (z axis), in degrees
        ang = np.degrees(np.arccos(np.clip(abs(n[2]), -1, 1)))
        tag = "HORIZONTAL(floor?)" if ang < 20 else \
              ("VERTICAL(wall?)" if ang > 70 else "TILTED")
        print("     trial %d: n=[%.3f %.3f %.3f]  %4.1f deg from vertical  "
              "inliers=%.1f%%  %s" %
              (i+1, n[0], n[1], n[2], ang, 100.0*ninl/ntot, tag))
        normals.append(n)
    normals = np.array(normals)
    # pairwise max angle between the trial normals
    max_spread = 0.0
    for i in range(len(normals)):
        for j in range(i+1, len(normals)):
            a = np.degrees(np.arccos(np.clip(abs(normals[i] @ normals[j]), -1, 1)))
            max_spread = max(max_spread, a)
    verdict = "STABLE" if max_spread < 5 else \
              ("UNSTABLE — RANSAC picking different planes!" if max_spread > 20
               else "somewhat unstable")
    print("     -> max spread between trials: %.1f deg  [%s]\n" %
          (max_spread, verdict))
    return normals[0]


def overlap_fitness(src_pts, tgt_kdtree, thresh):
    cnt = 0
    for p in src_pts:
        _, idx, d2 = tgt_kdtree.search_knn_vector_3d(p, 1)
        if d2[0] < thresh*thresh:
            cnt += 1
    return cnt / max(len(src_pts), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("target")
    ap.add_argument("--plane-trials", type=int, default=5)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    ref_path = os.path.expanduser(args.reference)
    tgt_path = os.path.expanduser(args.target)

    params = AlignParams()
    ref = o3d.io.read_point_cloud(ref_path)
    tgt = o3d.io.read_point_cloud(tgt_path)
    assert len(ref.points) and len(tgt.points), "empty cloud(s)"
    print("loaded ref=%d tgt=%d points\n" % (len(ref.points), len(tgt.points)))

    print("=== GROUND-PLANE STABILITY CHECK ===")
    print("Both should be HORIZONTAL(floor?) and STABLE. A VERTICAL/wall result,")
    print("or normals changing between trials, is the indoor failure mode.\n")
    ground_stability(ref, params, "REF", args.plane_trials)
    ground_stability(tgt, params, "TGT", args.plane_trials)

    print("=== ALIGNMENT ===")
    prep = PreparedReference(ref, params)
    T, info = align_target(tgt, prep, params, logger=lambda m: print("  "+m))
    print("\ninfo:", info)

    tgt_moved = o3d.geometry.PointCloud(tgt)
    tgt_moved.transform(T)
    tgt_moved_ds = tgt_moved.voxel_down_sample(0.10)
    ref_ds = ref.voxel_down_sample(0.10)
    kdt = o3d.geometry.KDTreeFlann(ref_ds)
    sp = np.asarray(tgt_moved_ds.points)
    print("\nOverlap fitness (fraction of target within X of a reference point):")
    for thr in (0.10, 0.25, 0.50, 1.00, 2.00):
        print("  within %.2fm : %.3f" % (thr, overlap_fitness(sp, kdt, thr)))

    if not args.no_viewer:
        ref_ds.paint_uniform_color([0.2, 0.5, 1.0])
        tgt_moved_ds.paint_uniform_color([1.0, 0.5, 0.1])
        print("\nViewer: blue=reference, orange=transformed target. "
              "Overlap = good.")
        o3d.visualization.draw_geometries([ref_ds, tgt_moved_ds])


if __name__ == "__main__":
    main()