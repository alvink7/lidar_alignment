#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_levelling.py

Interactive Open3D view of what alignment_core selects as the ground plane for
levelling, for BOTH the reference and target clouds, and the levelling outcome.

Two view modes (toggle with the space bar):

  RAW view (default):
     reference cloud  -> dim blue,   its ground points -> bright cyan
     target cloud     -> dim orange, its ground points -> bright red
     the fitted floor plane of each cloud is drawn as a translucent quad,
     plus a small normal arrow at the floor centroid.
     Each cloud sits at its own height, so you can see the floor offset.

  LEVELLED view:
     each cloud shifted down by its own ground_z (the pure Z-shift the
     pipeline applies). Both floors now coincide at Z=0 (drawn as a grey grid),
     so a correct selection makes the two ground bands sit flush on the plane.

Keys:  [space] toggle raw/levelled    [g] toggle ground highlight    [q] quit

Usage:
    python visualize_levelling.py ref.ply target.ply [--voxel 0.10]
"""
import argparse
import numpy as np
import open3d as o3d

from alignment_core import AlignParams, level_cloud, fit_floor_plane


# ----- colours -----
REF_DIM   = [0.20, 0.35, 0.75]
REF_HI    = [0.10, 0.90, 1.00]
TGT_DIM   = [0.80, 0.55, 0.20]
TGT_HI    = [1.00, 0.15, 0.15]


def load(path, params):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(params.ICP_VOXEL)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=params.NORMAL_RADIUS, max_nn=30))
    return pcd


def ground_mask(pts, floor_pts):
    """Boolean mask over pts marking the floor points the fit selected."""
    mask = np.zeros(len(pts), dtype=bool)
    if len(floor_pts) == 0:
        return mask
    kd = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    for p in floor_pts:
        _, idx, _ = kd.search_knn_vector_3d(p, 1)
        mask[idx[0]] = True
    return mask


def coloured_cloud(pts, mask, dim_col, hi_col, show_hi=True):
    c = np.tile(np.asarray(dim_col, float), (len(pts), 1))
    if show_hi:
        c[mask] = hi_col
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(c)
    return pc


def plane_quad(a, b, c, floor_pts, colour):
    """Translucent-ish quad z = a*x+b*y+c spanning the floor extent, as a
    thin mesh. Open3D legacy has no alpha, so we use a light tint."""
    if len(floor_pts) < 3:
        return None
    xs, ys = floor_pts[:, 0], floor_pts[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    verts = np.array([[x, y, a * x + b * y + c] for x, y in corners])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(tris)
    m.paint_uniform_color(colour)
    m.compute_vertex_normals()
    return m


def normal_arrow(a, b, c, floor_pts, colour, length=1.0):
    if len(floor_pts) == 0:
        return None
    cx, cy = float(floor_pts[:, 0].mean()), float(floor_pts[:, 1].mean())
    cz = a * cx + b * cy + c
    n = np.array([a, b, -1.0]); n /= np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    arr = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.03, cone_radius=0.07,
        cylinder_height=length * 0.75, cone_height=length * 0.25)
    arr.paint_uniform_color(colour)
    # rotate +Z -> n
    zaxis = np.array([0, 0, 1.0])
    v = np.cross(zaxis, n); s = np.linalg.norm(v); d = float(np.dot(zaxis, n))
    if s > 1e-9:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - d) / (s * s))
        arr.rotate(R, center=(0, 0, 0))
    arr.translate((cx, cy, cz))
    return arr


def ground_grid(size=8.0, step=0.5, z=0.0):
    """Grey wireframe grid on Z=z to mark the levelled floor plane."""
    lines, pts = [], []
    n = int(size / step)
    k = 0
    for i in range(-n, n + 1):
        x = i * step
        pts += [[x, -size, z], [x, size, z]]; lines.append([k, k + 1]); k += 2
        pts += [[-size, x, z], [size, x, z]]; lines.append([k, k + 1]); k += 2
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))
    ls.paint_uniform_color([0.45, 0.45, 0.45])
    return ls


def build(ref_path, tgt_path, params):
    # four precomputed geometry sets: (mode raw/lev) x (highlight on/off)
    geoms = {("raw", True): [], ("raw", False): [],
             ("lev", True): [], ("lev", False): []}
    report = {}

    for path, dim, hi, tag in ((ref_path, REF_DIM, REF_HI, "ref"),
                               (tgt_path, TGT_DIM, TGT_HI, "target")):
        pcd = load(path, params)
        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        a, b, c, floor_pts, n_floor, horiz = fit_floor_plane(pts, normals, params)
        _, T, meta = level_cloud(pcd, params)
        gz = meta["ground_z"]
        mask = ground_mask(pts, floor_pts)
        report[tag] = (n_floor, horiz, gz, meta["tilt_deg"])

        lev_pts = pts.copy(); lev_pts[:, 2] -= gz
        for show in (True, False):
            geoms[("raw", show)].append(
                coloured_cloud(pts, mask, dim, hi, show_hi=show))
            geoms[("lev", show)].append(
                coloured_cloud(lev_pts, mask, dim, hi, show_hi=show))

        # plane quad + normal arrow only in raw mode, only when highlight on
        q = plane_quad(a, b, c, floor_pts, hi)
        if q: geoms[("raw", True)].append(q)
        ar = normal_arrow(a, b, c, floor_pts, hi)
        if ar: geoms[("raw", True)].append(ar)

    grid = ground_grid(z=0.0)
    geoms[("lev", True)].append(grid)
    geoms[("lev", False)].append(grid)
    return geoms, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("target")
    ap.add_argument("--voxel", type=float, default=None)
    args = ap.parse_args()

    params = AlignParams()
    if args.voxel is not None:
        params.ICP_VOXEL = args.voxel

    geoms, report = build(args.ref, args.target, params)

    print("\n================ ground selection ================")
    for tag in ("ref", "target"):
        n_floor, horiz, gz, tilt = report[tag]
        print("%-7s floor_pts=%-5d horizontal_used=%-5s ground_z=%+.4f "
              "tilt=%.2fdeg" % (tag, n_floor, horiz, gz, tilt))
    dz = report["ref"][2] - report["target"][2]
    print("floor Z offset (ref - target) = %+.4f m" % dz)
    print("==================================================")
    print("\nWindow keys:  [space] raw<->levelled   [g] ground highlight   [q] quit")
    print("RAW:      ref=blue/cyan  target=orange/red  (bright = selected floor)")
    print("LEVELLED: both shifted to Z=0 grid; floor bands should sit on the grid\n")

    state = {"mode": "raw", "hi": True}

    def render(vis, reset=False):
        vis.clear_geometries()
        for g in geoms[(state["mode"], state["hi"])]:
            vis.add_geometry(g, reset_bounding_box=reset)
        vis.get_render_option().point_size = 2.5

    def toggle_mode(vis):
        state["mode"] = "lev" if state["mode"] == "raw" else "raw"
        render(vis)
        print("mode ->", state["mode"])
        return False

    def toggle_hi(vis):
        state["hi"] = not state["hi"]
        render(vis)
        print("ground highlight ->", state["hi"])
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("levelling: ground selection + outcome", 1280, 800)
    for g in geoms[("raw", True)]:
        vis.add_geometry(g)
    vis.get_render_option().point_size = 2.5
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.07])
    vis.register_key_callback(ord(" "), toggle_mode)
    vis.register_key_callback(ord("G"), toggle_hi)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()