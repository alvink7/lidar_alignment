#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_alignment.py
----------------------
Runs the alignment on two PCDs and shows the result overlaid:
  blue   = reference (fixed)
  orange = target, transformed into the reference frame by T_full

If the two colors sit on top of each other, the alignment is good — regardless
of what the TEASER/GICP fitness scalars say (they read low on sparse outdoor
LiDAR even when the transform is correct).

Usage:
  python3 visualize_alignment.py REF.pcd TGT.pcd
  python3 visualize_alignment.py REF.pcd TGT.pcd --voxel 0.10
  python3 visualize_alignment.py REF.pcd TGT.pcd --save aligned_target.pcd
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
from alignment_core import AlignParams, PreparedReference, align_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("target")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="downsample for display only (0 = full res)")
    ap.add_argument("--save", default="",
                    help="optional: write the transformed target to this PCD")
    args = ap.parse_args()

    ref_path = os.path.expanduser(args.reference)
    tgt_path = os.path.expanduser(args.target)

    params = AlignParams()
    ref = o3d.io.read_point_cloud(ref_path)
    tgt = o3d.io.read_point_cloud(tgt_path)
    assert len(ref.points) > 0, "reference empty/not found: " + ref_path
    assert len(tgt.points) > 0, "target empty/not found: " + tgt_path
    print("loaded ref=%d tgt=%d points" % (len(ref.points), len(tgt.points)))

    # --- run alignment ---
    prep = PreparedReference(ref, params)
    T, info = align_target(tgt, prep, params, logger=lambda m: print("  " + m))

    np.set_printoptions(precision=4, suppress=True)
    print("\nT (target -> reference):\n", T)
    print("\nfitness/rmse info:", info)

    # --- apply transform to a COPY of the full target ---
    tgt_moved = o3d.geometry.PointCloud(tgt)   # copy
    tgt_moved.transform(T)

    # --- downsample for display ---
    if args.voxel > 0:
        ref_show = ref.voxel_down_sample(args.voxel)
        tgt_show = tgt_moved.voxel_down_sample(args.voxel)
    else:
        ref_show, tgt_show = ref, tgt_moved

    ref_show.paint_uniform_color([0.20, 0.50, 1.00])   # blue
    tgt_show.paint_uniform_color([1.00, 0.50, 0.10])   # orange

    print("\nViewer: blue = reference, orange = transformed target.")
    print("Overlapping colors => good alignment. Close window to exit.")
    o3d.visualization.draw_geometries(
        [ref_show, tgt_show],
        window_name="Alignment: blue=ref orange=target",
        width=1280, height=800)


if __name__ == "__main__":
    main()