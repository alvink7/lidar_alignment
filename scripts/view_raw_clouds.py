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

  # overlay each run's FAST-LIO trajectory on its own cloud (one source per PCD, same order)
  python3 view_raw_clouds.py ref.pcd tgt.pcd --odom ref.bag tgt.bag

  --odom takes a .bag (reads --odom-topic, default /Odometry) or a TUM text file
  (t x y z qx qy qz qw). A cloud and its odometry should share a frame: the first
  pose sits at the origin and the path stays in free space inside the scan.

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


def read_odom(path, topic):
    """(N,3) positions and first-pose (pos, quat_xyzw) from a bag or TUM file."""
    path = os.path.expanduser(path)
    if path.endswith(".bag"):
        import rosbag
        P, Q = [], []
        with rosbag.Bag(path) as bag:
            for _, msg, _ in bag.read_messages(topics=[topic]):
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                P.append([p.x, p.y, p.z])
                Q.append([q.x, q.y, q.z, q.w])
        if not P:
            sys.exit("ERROR: no %s messages in %s" % (topic, path))
        return np.asarray(P), np.asarray(Q[0])
    a = np.loadtxt(path, ndmin=2)
    return a[:, 1:4], a[0, 4:8]


def quat_angle_deg(q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    return float(np.degrees(2 * np.arccos(np.clip(abs(q[3]), 0, 1))))


def odom_check(name, cloud, traj, q0):
    print("%s odometry\n  poses    %d\n  first pose at (%.2f, %.2f, %.2f): %.2f m from origin, "
          "rotated %.1f deg from identity"
          % (name, len(traj), *traj[0], np.linalg.norm(traj[0]), quat_angle_deg(q0)))
    warn = []
    if np.linalg.norm(traj[0]) > 0.2 or quat_angle_deg(q0) > 5:
        warn.append("first pose is NOT at the origin/identity - this odometry frame is not "
                    "the drone's start pose (global or re-used frame?)")
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(cloud).query(traj)
        print("  clearance to nearest cloud point: min %.2f m, median %.2f m"
              % (d.min(), np.median(d)))
        if np.median(d) < 0.15:
            warn.append("trajectory runs through the cloud (median clearance %.2f m) - "
                        "cloud and odometry are probably in different frames" % np.median(d))
        far = np.median(d) > 3.0
        if far:
            warn.append("trajectory is far from every cloud point (median %.1f m) - "
                        "cloud and odometry are probably in different frames" % np.median(d))
    except ImportError:
        pass
    for w in warn:
        print("  WARNING:", w)
    if not warn:
        print("  OK: starts at the origin and stays in free space")


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


def darker(h, f=0.55):
    return [c * f for c in hex_rgb(h)]


def show_open3d(clouds, axis_size, point_size, trajs=()):
    import open3d as o3d
    geoms = []
    for (name, p), col in zip(clouds, COLORS):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        pc.paint_uniform_color(hex_rgb(col))
        geoms.append(pc)
    for traj, col in zip(trajs, COLORS):
        if traj is None:
            continue
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(traj),
                                  o3d.utility.Vector2iVector(
                                      [[k, k + 1] for k in range(len(traj) - 1)]))
        ls.paint_uniform_color(darker(col))
        geoms.append(ls)
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sph.translate(traj[0])
        sph.paint_uniform_color(darker(col))
        geoms.append(sph)
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


def save_png(clouds, path, trajs=(), max_pts=150000):
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
        for traj, col in zip(trajs, COLORS):
            if traj is None:
                continue
            ax.plot(traj[:, i], traj[:, j], color="white", lw=3.5, zorder=3)
            ax.plot(traj[:, i], traj[:, j], color=col, lw=1.8, zorder=4)
            ax.plot(traj[0, i], traj[0, j], "o", color=col, mec="white", mew=1.5,
                    ms=9, zorder=6)
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
    if any(t is not None for t in trajs):
        handles.append(Line2D([], [], color="#52514e", lw=1.8, marker="o", ms=7,
                              label="odometry path (dot = start)"))
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
    ap.add_argument("--odom", nargs="+", default=[],
                    help="odometry per cloud, same order (.bag or TUM file; '-' to skip one)")
    ap.add_argument("--odom-topic", default="/Odometry")
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

    if len(args.odom) > len(clouds):
        sys.exit("ERROR: more --odom sources than clouds")
    trajs = []
    for k, src in enumerate(args.odom):
        if src == "-":
            trajs.append(None)
            continue
        traj, q0 = read_odom(src, args.odom_topic)
        print()
        odom_check(clouds[k][0], clouds[k][1], traj, q0)
        trajs.append(traj)

    if args.png:
        save_png(clouds, args.png, trajs)
        return
    try:
        show_open3d(clouds, args.axis_size, args.point_size, trajs)
    except ImportError:
        out = "raw_clouds.png"
        print("open3d not installed - saving %s instead" % out)
        save_png(clouds, out, trajs)


if __name__ == "__main__":
    main()
