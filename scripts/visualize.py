#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_initial_frames.py
---------------------
Visualize the REFERENCE and TARGET point clouds in their ORIGINAL (native)
frames — no levelling, no alignment, no transforms applied. Just loads both
PCDs as-is and draws them together so you can see how they sit relative to each
other straight out of FAST-LIO.

Reference is drawn in one colour, target in another, plus a coordinate axis at
the origin so you can see each cloud's own frame orientation.

Usage:
    python3 viz_initial_frames.py REF.pcd TGT.pcd
    python3 viz_initial_frames.py REF.pcd TGT.pcd --voxel 0.1
    python3 viz_initial_frames.py REF.pcd TGT.pcd --no-axes --point-size 2
"""

import argparse
import sys
import numpy as np
import open3d as o3d


REF_COLOR = [0.90, 0.30, 0.20]   # warm red  = reference
TGT_COLOR = [0.20, 0.55, 0.90]   # blue      = target


def load_cloud(path, voxel):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        sys.exit("ERROR: %s is empty or failed to load." % path)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def describe(label, pcd):
    pts = np.asarray(pcd.points)
    ctr = pts.mean(axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    print("  [%s] %d pts | center=(%.2f, %.2f, %.2f) | "
          "extent=(%.2f, %.2f, %.2f)" %
          (label, len(pts), ctr[0], ctr[1], ctr[2],
           mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]))


def main():
    ap = argparse.ArgumentParser(
        description="Visualize two point clouds in their raw initial frames.")
    ap.add_argument("reference", help="reference .pcd path")
    ap.add_argument("target", help="target .pcd path")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="optional voxel size for downsampling (0 = none)")
    ap.add_argument("--point-size", type=float, default=1.5,
                    help="render point size")
    ap.add_argument("--no-axes", action="store_true",
                    help="hide the origin coordinate axes")
    ap.add_argument("--uniform-color", action="store_true",
                    help="force flat colors even if the clouds have their own")
    args = ap.parse_args()

    print("Loading clouds (NO transforms — native frames):")
    ref = load_cloud(args.reference, args.voxel)
    tgt = load_cloud(args.target, args.voxel)

    describe("REF", ref)
    describe("TGT", tgt)

    # colour them so they're distinguishable
    if args.uniform_color or not ref.has_colors():
        ref.paint_uniform_color(REF_COLOR)
    if args.uniform_color or not tgt.has_colors():
        tgt.paint_uniform_color(TGT_COLOR)

    geoms = [ref, tgt]

    if not args.no_axes:
        # size the axes to the larger cloud so they're visible
        span = np.linalg.norm(
            np.asarray(ref.get_axis_aligned_bounding_box().get_extent()))
        axis_size = max(0.5, span * 0.1)
        geoms.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))

    print("\nRED = reference   BLUE = target   (both in their OWN frames)")
    print("Close the window to exit.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Initial frames: REF (red) vs TGT (blue)")
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()