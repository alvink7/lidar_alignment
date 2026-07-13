#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
level_clouds.py — standalone ground-plane levelling check.

Loads two PCDs (reference, target), levels each with the SAME level_cloud() /
fit_floor_plane() the pipeline uses, and shows you exactly what it did:
  * which points it selected as FLOOR (highlighted green),
  * the fitted plane NORMAL, its tilt from +Z, and whether the gate rotated
    the cloud or fell back to Z-only,
  * before/after views so you can confirm each cloud sits flat on Z=0 and the
    two floors end up coplanar.

Get THIS working (floors flat, normals ~[0,0,1], both coplanar at Z=0) before
trusting the TEASER/GICP stage. If the two clouds still disagree in yaw after
both are correctly levelled, the problem is global registration, not levelling.

Usage:
    python3 level_clouds.py REFERENCE.pcd TARGET.pcd
    python3 level_clouds.py REFERENCE.pcd TARGET.pcd --no-vis        # print only
    python3 level_clouds.py REFERENCE.pcd TARGET.pcd --save out_dir  # write PCDs
    python3 level_clouds.py REFERENCE.pcd TARGET.pcd --clean         # drop outliers

Both clouds must be gravity-aligned (stock FAST-LIO /cloud_registered).
"""
import argparse
import copy
import os
import sys

import numpy as np
import open3d as o3d

from alignment_core import AlignParams, fit_floor_plane, level_cloud

BLUE   = [0.20, 0.55, 0.95]   # reference
ORANGE = [1.00, 0.55, 0.10]   # target
GREEN  = [0.10, 0.85, 0.30]   # selected floor points


def load_and_prep(path, params, clean=False):
    pcd = o3d.io.read_point_cloud(path)
    n0 = len(pcd.points)
    if n0 == 0:
        sys.exit(f"ERROR: empty or unreadable PCD: {path}")
    down = pcd.voxel_down_sample(params.ICP_VOXEL)
    if clean:
        down, _ = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=params.NORMAL_RADIUS, max_nn=30))
    print(f"  loaded {os.path.basename(path)}: {n0} -> {len(down.points)} pts"
          f"{' (outliers removed)' if clean else ''}")
    return down


def floor_cloud(floor_pts):
    fc = o3d.geometry.PointCloud()
    if len(floor_pts):
        fc.points = o3d.utility.Vector3dVector(np.asarray(floor_pts))
    fc.paint_uniform_color(GREEN)
    return fc


def analyze(name, pcd, params):
    """Print the fit and return (floor_pts, meta) without modifying pcd."""
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    a, b, c, floor_pts, n_floor, horiz = fit_floor_plane(pts, normals, params)

    normal = np.array([a, b, -1.0])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))

    cx = float(np.mean(floor_pts[:, 0])) if len(floor_pts) else 0.0
    cy = float(np.mean(floor_pts[:, 1])) if len(floor_pts) else 0.0
    ground_z = float(a * cx + b * cy + c) if len(floor_pts) else float(c)

    z = pts[:, 2]
    print(f"\n[{name}]")
    print(f"  floor points selected : {n_floor}  (normal filter used: {horiz})")
    print(f"  fitted normal         : [{normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f}]")
    print(f"  tilt from +Z          : {tilt:.2f} deg  (diagnostic only -- NOT applied; "
          f"levelling is pure Z-shift)")
    print(f"  floor height (Z-shift) : {ground_z:+.2f} m")
    print(f"  Z range before        : {z.min():+.2f} -> {z.max():+.2f} m")
    return floor_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("target")
    ap.add_argument("--no-vis", action="store_true", help="print only, no windows")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write levelled clouds to this directory")
    ap.add_argument("--clean", action="store_true",
                    help="remove statistical outliers (e.g. the drift 'arm')")
    args = ap.parse_args()

    params = AlignParams()
    print("Loading + downsampling (%.2fm voxel) ..." % params.ICP_VOXEL)
    ref = load_and_prep(args.reference, params, clean=args.clean)
    tgt = load_and_prep(args.target, params, clean=args.clean)

    print("\n=== GROUND-PLANE FIT ===")
    ref_floor = analyze("REFERENCE (blue)", ref, params)
    tgt_floor = analyze("TARGET (orange)", tgt, params)

    # Level each with the real pipeline function
    ref_lev, T_ref, ref_meta = level_cloud(ref, params)
    tgt_lev, T_tgt, tgt_meta = level_cloud(tgt, params)

    rz = np.asarray(ref_lev.points)[:, 2]
    tz = np.asarray(tgt_lev.points)[:, 2]
    print("\n=== AFTER LEVELLING (floor should sit at Z~0) ===")
    print(f"  reference Z range : {rz.min():+.2f} -> {rz.max():+.2f} m")
    print(f"  target    Z range : {tz.min():+.2f} -> {tz.max():+.2f} m")
    dot = float(np.clip(np.dot(ref_meta["normal"], tgt_meta["normal"]), -1, 1))
    print(f"  angle between the two fitted floor normals: "
          f"{np.degrees(np.arccos(dot)):.2f} deg  (want ~0)")
    np.set_printoptions(precision=4, suppress=True)
    print("\n  T_ref_level =\n", T_ref, "\n  T_tgt_level =\n", T_tgt)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(args.save, "ref_levelled.pcd"), ref_lev)
        o3d.io.write_point_cloud(os.path.join(args.save, "tgt_levelled.pcd"), tgt_lev)
        print(f"\n  wrote ref_levelled.pcd / tgt_levelled.pcd to {args.save}")

    if args.no_vis:
        return

    def show(geoms, title):
        try:
            o3d.visualization.draw_geometries(geoms, window_name=title,
                                              width=1280, height=720)
        except Exception as e:
            print(f"  (visualisation unavailable: {e} -- use --save and an "
                  f"external viewer)")

    # 1) what got picked as floor, per cloud (full cloud + green floor points)
    r = copy.deepcopy(ref); r.paint_uniform_color(BLUE)
    show([r, floor_cloud(ref_floor)], "REFERENCE — green = selected floor")
    t = copy.deepcopy(tgt); t.paint_uniform_color(ORANGE)
    show([t, floor_cloud(tgt_floor)], "TARGET — green = selected floor")

    # 2) after levelling, both clouds + origin axis (floors should meet at Z=0)
    rl = copy.deepcopy(ref_lev); rl.paint_uniform_color(BLUE)
    tl = copy.deepcopy(tgt_lev); tl.paint_uniform_color(ORANGE)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    show([rl, tl, axis], "AFTER LEVELLING — both floors should sit on Z=0")


if __name__ == "__main__":
    main()