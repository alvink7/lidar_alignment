#!/usr/bin/env python3
"""
view_raw_clouds.py - show PCD files in their own raw frames, no transforms applied.

Each cloud gets its own colour, and a coordinate frame is drawn at the shared
origin (0,0,0), so you see exactly how the clouds sit relative to each other
before any alignment: e.g. where each FAST-LIO run's camera_init is.

Usage
-----
  python3 view_raw_clouds.py ref.pcd tgt.pcd                 # interactive 3D window
  python3 view_raw_clouds.py ref.pcd tgt.pcd --voxel 0.1     # downsample big clouds
  python3 view_raw_clouds.py ref.pcd tgt.pcd --png raw.png   # no GUI: top-down + side image

Interactive controls (Open3D): left-drag rotate, scroll zoom, shift+drag pan,
+/- point size, R reset view, Q quit.
"""

import argparse
import os
import sys

import numpy as np

# categorical colours, fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def read_pcd(path):
    """xyz array. Uses open3d if installed, else a minimal ascii/binary PCD reader."""
    try:
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(path).points, dtype=float)
    except ImportError:
        pass
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline()
            if not line:
                sys.exit("ERROR: %s: no DATA line in header" % path)
            line = line.decode("ascii", "ignore").strip()
            if not line or line.startswith("#"):
                continue
            k, *v = line.split()
            hdr[k.upper()] = v
            if k.upper() == "DATA":
                break
        fields, n = hdr["FIELDS"], int(hdr["POINTS"][0])
        if hdr["DATA"][0] == "ascii":
            a = np.loadtxt(f, ndmin=2)
            return a[:, [fields.index(c) for c in "xyz"]]
        if hdr["DATA"][0] != "binary":
            sys.exit("ERROR: %s is %s - install open3d to read it" % (path, hdr["DATA"][0]))
        types = {("F", "4"): "f4", ("F", "8"): "f8", ("U", "1"): "u1", ("U", "2"): "u2",
                 ("U", "4"): "u4", ("I", "1"): "i1", ("I", "2"): "i2", ("I", "4"): "i4"}
        counts = hdr.get("COUNT", ["1"] * len(fields))
        dt = np.dtype([(fn, "<" + types[(ty, sz)], (int(c),) if int(c) > 1 else ())
                       for fn, ty, sz, c in zip(fields, hdr["TYPE"], hdr["SIZE"], counts)])
        a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([a["x"], a["y"], a["z"]], 1).astype(float)


def voxel_down(p, voxel):
    if voxel <= 0:
        return p
    keys = np.floor(p / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return p[np.sort(idx)]


def describe(name, p):
    lo, hi = p.min(0), p.max(0)
    c = p.mean(0)
    print("%s\n  points   %d\n  centroid (%.2f, %.2f, %.2f)\n  min      (%.2f, %.2f, %.2f)\n"
          "  max      (%.2f, %.2f, %.2f)\n  extent   %.2f x %.2f x %.2f m\n"
          "  centroid distance from origin %.2f m"
          % (name, len(p), *c, *lo, *hi, *(hi - lo), np.linalg.norm(c)))


def show_open3d(clouds, axis_size, point_size):
    import open3d as o3d
    geoms = []
    for (name, p), col in zip(clouds, COLORS):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        pc.paint_uniform_color(hex_rgb(col))
        geoms.append(pc)
    if axis_size > 0:
        # red = x, green = y, blue = z
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))
    vis = o3d.visualization.Visualizer()
    vis.create_window("Raw clouds (no transforms)", 1400, 900)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([1.0, 1.0, 1.0])
    vis.run()
    vis.destroy_window()


def save_png(clouds, path, max_pts=150000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (i, j, xl, yl, title) in zip(axes, [(0, 1, "x (m)", "y (m)", "Top-down (x-y)"),
                                                (0, 2, "x (m)", "z (m)", "Side (x-z)")]):
        for (name, p), col in zip(clouds, COLORS):
            q = p[rng.choice(len(p), max_pts, replace=False)] if len(p) > max_pts else p
            ax.scatter(q[:, i], q[:, j], s=0.3, c=col, alpha=0.5, linewidths=0, rasterized=True)
        ax.plot(0, 0, "+", color="#0b0b0b", ms=16, mew=2, zorder=5)
        ax.annotate("origin", (0, 0), xytext=(6, 6), textcoords="offset points",
                    color="#0b0b0b", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xl, color="#52514e")
        ax.set_ylabel(yl, color="#52514e")
        ax.set_title(title, loc="left", color="#0b0b0b")
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#a9a8a2")
        ax.tick_params(colors="#52514e", labelsize=9)
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=8,
                      label="%s (%d pts)" % (n, len(p))) for (n, p), c in zip(clouds, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(clouds), 4), frameon=False)
    fig.suptitle("Raw clouds, each in its own frame (no transforms)", x=0.01, ha="left",
                 color="#0b0b0b", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    print("Saved %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcds", nargs="+", help="PCD files (up to 8)")
    ap.add_argument("--voxel", type=float, default=0.0, help="downsample voxel size (m)")
    ap.add_argument("--axis-size", type=float, default=1.0,
                    help="origin frame size in m (0 hides it)")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--png", help="save a top-down + side image instead of opening a window")
    args = ap.parse_args()
    if len(args.pcds) > len(COLORS):
        sys.exit("ERROR: at most %d clouds" % len(COLORS))

    clouds = []
    for path in args.pcds:
        path = os.path.expanduser(path)
        p = read_pcd(path)
        p = p[np.all(np.isfinite(p), axis=1)]
        if len(p) == 0:
            sys.exit("ERROR: no points in %s" % path)
        p = voxel_down(p, args.voxel)
        clouds.append((os.path.basename(path), p))

    names = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]
    for (name, p), cname in zip(clouds, names):
        describe("%s  [%s]" % (name, cname), p)
    if len(clouds) >= 2:
        d = clouds[1][1].mean(0) - clouds[0][1].mean(0)
        print("\nCentroid offset %s -> %s: (%.2f, %.2f, %.2f), %.2f m"
              % (clouds[0][0], clouds[1][0], *d, np.linalg.norm(d)))

    if args.png:
        save_png(clouds, args.png)
        return
    try:
        show_open3d(clouds, args.axis_size, args.point_size)
    except ImportError:
        out = "raw_clouds.png"
        print("open3d not installed - saving %s instead" % out)
        save_png(clouds, out)


if __name__ == "__main__":
    main()
