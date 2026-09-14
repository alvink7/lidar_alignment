#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_transform.py
------------------
Paste a 4x4 transform (numpy's own print format, brackets and all) and see
what it does to the test target cloud against the reference. No ROS needed --
just edit the TRANSFORM string below (or pass --matrix / --matrix-file) and
run.

Why a string parser: the matrix comes out of logs, rospy.loginfo dumps,
np.array2string, etc. already formatted like

    [[ 0.95628644  0.06226999 -0.28572485 14.28591555]
     [-0.21918224  0.79942792 -0.55935154 12.35018126]
     [ 0.19358561  0.59752611  0.77813056 -9.07000091]
     [ 0.          0.          0.          1.        ]]

Re-typing that as a clean np.array(...) literal every time is busywork and
error-prone. This just strips the brackets and whitespace-splits, so you can
paste it verbatim.

USAGE
    python3 test_transform.py
    python3 test_transform.py --matrix "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
    python3 test_transform.py --matrix-file T.txt
    python3 test_transform.py --target ~/maps/other.pcd --no-window
"""

import argparse
import os
import re
import sys

import numpy as np
import open3d as o3d


# ---- edit this and run with no args to test a transform quickly -----------
TRANSFORM = """
[[ 0.026  0.997  0.077 -0.834]
 [-0.998  0.031 -0.059  0.987]
 [-0.061 -0.075  0.995  0.582]
 [ 0.     0.     0.     1.   ]]
"""

REF_COLOR = [0.90, 0.30, 0.20]   # warm red  = reference
TGT_COLOR = [0.20, 0.55, 0.90]   # blue      = target, transformed

DEFAULT_REF = "~/catkin_ws/maps/reference_jul1.pcd"
DEFAULT_TGT = "~/catkin_ws/maps/test_target.pcd"


def parse_matrix(text):
    """Parse a 4x4 matrix out of numpy's print format (or anything close to
    it): strip brackets, pull out all the floats in reading order, reshape."""
    cleaned = re.sub(r"[\[\],]", " ", text)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if len(nums) != 16:
        sys.exit("ERROR: found %d numbers in the matrix text, expected 16.\n"
                  "Text was:\n%s" % (len(nums), text))
    T = np.array([float(n) for n in nums], dtype=np.float64).reshape(4, 4)
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6):
        print("WARNING: bottom row is %s, not [0 0 0 1] -- this may not be "
              "a valid rigid/affine transform." % T[3, :].tolist())
    if not np.all(np.isfinite(T)):
        sys.exit("ERROR: parsed matrix has NaN/Inf values:\n%s" % T)
    return T


def load_cloud(path, voxel):
    path = os.path.expanduser(path)
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
        description="Apply a pasted 4x4 transform to the test target cloud "
                    "and view it against the reference.")
    ap.add_argument("--reference", default=DEFAULT_REF, help="reference .pcd")
    ap.add_argument("--target", default=DEFAULT_TGT, help="target .pcd")
    ap.add_argument("--matrix", default=None,
                    help="4x4 matrix as text, overrides the TRANSFORM "
                         "constant in the script")
    ap.add_argument("--matrix-file", default=None,
                    help="read the matrix text from a file instead")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="optional voxel size for downsampling (0 = none)")
    ap.add_argument("--point-size", type=float, default=1.5)
    ap.add_argument("--out", default=None,
                    help="write the transformed+colored overlay to this "
                         ".pcd instead of / in addition to viewing it")
    ap.add_argument("--no-window", action="store_true",
                    help="skip the viewer, just parse + report + optionally "
                         "write --out")
    args = ap.parse_args()

    if args.matrix_file:
        with open(args.matrix_file) as f:
            text = f.read()
    elif args.matrix:
        text = args.matrix
    else:
        text = TRANSFORM

    T = parse_matrix(text)
    print("Parsed transform:\n%s\n" % T)

    ref = load_cloud(args.reference, args.voxel)
    tgt = load_cloud(args.target, args.voxel)

    print("Before:")
    describe("REF", ref)
    describe("TGT (raw)", tgt)

    tgt.transform(T)

    print("After applying transform to TGT:")
    describe("TGT (transformed)", tgt)

    ref.paint_uniform_color(REF_COLOR)
    tgt.paint_uniform_color(TGT_COLOR)

    if args.out:
        overlay = ref + tgt
        out_path = os.path.expanduser(args.out)
        o3d.io.write_point_cloud(out_path, overlay)
        print("Wrote overlay to %s" % out_path)

    if args.no_window:
        return

    print("\nRED = reference   BLUE = target (transformed)")
    print("Close the window to exit.")

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name="test_transform: REF (red) vs T*TGT (blue)")
    if not ok:
        sys.exit("ERROR: could not create a viewer window (no DISPLAY?). "
                  "Re-run with --out overlay.pcd --no-window instead.")
    vis.add_geometry(ref)
    vis.add_geometry(tgt)
    axis_size = max(0.5, np.linalg.norm(
        np.asarray(ref.get_axis_aligned_bounding_box().get_extent())) * 0.1)
    vis.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
