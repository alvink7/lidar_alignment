#!/usr/bin/env python3
"""
evaluate_alignment.py - prove cloud_aligner's transform against mocap ground truth.

Assumes: the REFERENCE flight moves, the TARGET is stationary.

Frames
------
  A = reference run's FAST-LIO frame (camera_init)    i = FAST-LIO body (IMU)
  B = target run's FAST-LIO frame (camera_init)       m = mocap rigid body (markers)
  M = mocap world frame, shared by both recordings
  X = T_i_m, fixed marker-to-IMU extrinsic (unknown - estimated here)

Method
------
  1. Reference (moving): fit T_M_A from the LIO trajectory vs mocap, and jointly
     estimate X from the per-timestamp difference between mocap and LIO poses:
        T_M_m(t) = T_M_A . T_A_i(t) . X
     Its per-sample consistency is a built-in check on the ground truth.
  2. Target (still): average the mocap pose and the LIO pose over the recording:
        T_M_B = T_M_m . inv(X) . inv(T_B_i)
  3. Ground truth:  T_A_B_gt = inv(T_M_A) . T_M_B
  4. Compare T_est (and the identity, as a do-nothing baseline) to T_A_B_gt:
       - rotation error (total / tilt / yaw, in the mocap frame)
       - translation error at the target drone's position
       - with --tgt-pcd: per-point placement error of the target cloud
         || T_est p - T_gt p ||  (how wrong the merged map is)

Outputs (in --out)
------------------
  summary.md                    presentation-ready tables + PASS/FAIL
  summary.json                  every number
  fig_error_comparison.png      aligner vs identity baseline vs thresholds
  fig_cloud_error.png           target cloud coloured by placement error   (--tgt-pcd)
  fig_cloud_error_cdf.png       distribution of placement error            (--tgt-pcd)
  fig_ground_truth_quality.png  reference fit, extrinsic consistency

Usage
-----
  python3 evaluate_alignment.py --ref-bag ref.bag --tgt-bag tgt.bag \
      --T T_est.txt --tgt-pcd Oct28_tgt.pcd --out eval_run1

  --T accepts .npy or a text file with 16 numbers (numpy's printed form is fine).
  --T-bag reads the node's /target_to_reference_matrix from a bag instead.

IMPORTANT: the reference /Odometry must come from the same FAST-LIO run that
produced the reference PCD, and the target /Odometry from the run the node
aligned (same camera_init frames as T_est).
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

if not hasattr(Rot, "mean"):
    sys.exit("ERROR: scipy is too old (need >= 1.4; Ubuntu 20.04's apt version is 1.3).\n"
             "  pip3 install --user --upgrade 'scipy>=1.4' 'matplotlib>=3.5'")

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _stamp(msg, t_bag):
    h = getattr(msg, "header", None)
    if h is not None and hasattr(h, "stamp"):
        s = h.stamp.to_sec()
        if s > 0:
            return s
    return t_bag.to_sec()


def _get(obj, path):
    for p in path.split("."):
        if not hasattr(obj, p):
            return None
        obj = getattr(obj, p)
    return obj


def _pose(msg):
    """(xyz, quat_xyzw) from hiperlab estimator/mocap output or standard pose msgs."""
    if all(hasattr(msg, f) for f in ("posx", "posy", "posz", "attq0")):
        # hiperlab: attq0 = w, attq1..3 = x, y, z
        return ([msg.posx, msg.posy, msg.posz],
                [msg.attq1, msg.attq2, msg.attq3, msg.attq0])
    for base in ("pose.pose", "pose", "transform"):
        o = _get(msg, base)
        if o is None:
            continue
        pos = getattr(o, "position", None) or getattr(o, "translation", None)
        ori = getattr(o, "orientation", None) or getattr(o, "rotation", None)
        if pos is not None and ori is not None:
            return [pos.x, pos.y, pos.z], [ori.x, ori.y, ori.z, ori.w]
    raise ValueError("Can't read a pose from %s (fields: %s)"
                     % (msg._type, msg.__slots__))


def load_poses(bag_path, topic, skip_zero=True):
    import rosbag
    t, p, q = [], [], []
    with rosbag.Bag(bag_path) as bag:
        for _, msg, tb in bag.read_messages(topics=[topic]):
            xyz, quat = _pose(msg)
            if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(quat))):
                continue
            if skip_zero and not any(xyz):
                continue                                    # mocap dropout
            if np.linalg.norm(quat) < 0.5:
                continue
            t.append(_stamp(msg, tb))
            p.append(xyz)
            q.append(quat)
    if not t:
        sys.exit("ERROR: no usable pose messages on %s in %s" % (topic, bag_path))
    t = np.asarray(t)
    o = np.argsort(t)
    q = np.asarray(q, float)[o]
    return {"t": t[o], "p": np.asarray(p, float)[o],
            "R": Rot.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))}


def load_T(args):
    if args.T:
        if args.T.endswith(".npy"):
            T = np.load(args.T)
        else:
            with open(args.T) as f:
                txt = f.read()
            for ch in "[](),;":
                txt = txt.replace(ch, " ")
            T = np.asarray([float(x) for x in txt.split()])
        T = np.asarray(T, float).reshape(4, 4)
    else:
        import rosbag
        T = None
        with rosbag.Bag(args.T_bag) as bag:
            for _, msg, _ in bag.read_messages(topics=[args.T_topic]):
                T = np.asarray(msg.data, float).reshape(4, 4)
        if T is None:
            sys.exit("ERROR: no %s message in %s" % (args.T_topic, args.T_bag))
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6) or abs(np.linalg.det(R) - 1) > 1e-3
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-3)):
        sys.exit("ERROR: T_est is not a valid rigid transform:\n%s" % T)
    return T


def load_pcd_xyz(path, max_points=200000, seed=0):
    path = os.path.expanduser(path)
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(path).points)
    except ImportError:
        pts = _read_pcd_fallback(path)
    if len(pts) == 0:
        sys.exit("ERROR: no points read from %s" % path)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


def _read_pcd_fallback(path):
    """Minimal PCD reader (ascii / binary, not binary_compressed)."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if not line or line.startswith("#"):
                continue
            k, *v = line.split()
            hdr[k.upper()] = v
            if k.upper() == "DATA":
                break
        fields = hdr["FIELDS"]
        n = int(hdr["POINTS"][0])
        ix = [fields.index(c) for c in "xyz"]
        if hdr["DATA"][0] == "ascii":
            a = np.loadtxt(f, ndmin=2)
            return a[:, ix]
        if hdr["DATA"][0] != "binary":
            sys.exit("ERROR: %s is %s; install open3d to read it" % (path, hdr["DATA"][0]))
        types = {("F", "4"): "f4", ("F", "8"): "f8", ("U", "1"): "u1", ("U", "2"): "u2",
                 ("U", "4"): "u4", ("I", "1"): "i1", ("I", "2"): "i2", ("I", "4"): "i4"}
        counts = hdr.get("COUNT", ["1"] * len(fields))
        dt = np.dtype([(fn, "<" + types[(ty, sz)], (int(c),) if int(c) > 1 else ())
                       for fn, ty, sz, c in zip(fields, hdr["TYPE"], hdr["SIZE"], counts)])
        a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([a["x"], a["y"], a["z"]], 1).astype(float)


# --------------------------------------------------------------------------- #
# SE(3) helpers
# --------------------------------------------------------------------------- #

def mk(R, t):
    T = np.eye(4)
    T[:3, :3] = R.as_matrix() if isinstance(R, Rot) else R
    T[:3, 3] = t
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def stack(poses):
    """poses dict -> (N,4,4)."""
    n = len(poses["t"])
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = poses["R"].as_matrix()
    T[:, :3, 3] = poses["p"]
    return T


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, _, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    spread = np.sqrt(np.linalg.svd(S, compute_uv=False) ** 2 / len(src))
    return mk(R, mu_d - R @ mu_s), spread


def mean_pose(Ts):
    R = Rot.from_matrix(Ts[:, :3, :3]).mean()
    return mk(R, Ts[:, :3, 3].mean(0))


def rot_deg(R):
    R = R[:3, :3] if R.shape == (4, 4) else R
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def tilt_yaw_deg(R, up):
    tilt = float(np.degrees(np.arccos(np.clip((R @ up) @ up, -1, 1))))
    h = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0])
    h /= np.linalg.norm(h)
    hr = R @ h
    hr -= (hr @ up) * up
    yaw = float(np.degrees(np.arctan2(np.cross(h, hr) @ up, h @ hr)))
    return tilt, yaw


def associate(t_query, ref, max_gap, t_offset=0.0):
    """Interpolate ref poses (lerp position, slerp rotation) at t_query + offset."""
    tq = t_query + t_offset
    tr = ref["t"]
    i = np.searchsorted(tr, tq)
    ok = (i > 0) & (i < len(tr))
    i = np.clip(i, 1, len(tr) - 1)
    t0, t1 = tr[i - 1], tr[i]
    ok &= (t1 - t0) <= max_gap
    w = np.clip(np.where(t1 > t0, (tq - t0) / np.maximum(t1 - t0, 1e-12), 0.0), 0, 1)
    p = (1 - w)[:, None] * ref["p"][i - 1] + w[:, None] * ref["p"][i]
    R0, R1 = ref["R"][i - 1], ref["R"][i]
    R = R0 * Rot.from_rotvec((R0.inv() * R1).as_rotvec() * w[:, None])
    return ok, {"t": t_query, "p": p, "R": R}


def subset(poses, mask):
    return {"t": poses["t"][mask], "p": poses["p"][mask], "R": poses["R"][mask]}


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

def fit_reference(T_A_i, T_M_m, iters=10):
    """Jointly estimate T_M_A and X = T_i_m from a moving flight.
    T_M_m(t) = T_M_A . T_A_i(t) . X"""
    X = np.eye(4)
    R_A_i, p_A = T_A_i[:, :3, :3], T_A_i[:, :3, 3]
    R_M_m, p_m = T_M_m[:, :3, :3], T_M_m[:, :3, 3]
    for _ in range(iters):
        # rotation of A in mocap: position fit of IMU positions (via current X)
        p_imu_M = np.einsum("nij,j->ni", T_M_m @ inv(X), [0, 0, 0, 1])[:, :3]
        T_M_A, spread = umeyama(p_A, p_imu_M)
        R_M_A = T_M_A[:3, :3]
        # extrinsic rotation: R_M_m = R_M_A R_A_i R_X
        R_X = Rot.from_matrix(np.einsum("nji,jk,nkl->nil", R_A_i, R_M_A.T, R_M_m)).mean()
        # lever arm l and frame offset t jointly (linear):
        #   p_m = R_M_A p_A + t + R_M_A R_A_i l
        J = np.concatenate([np.tile(np.eye(3), (len(p_A), 1)),
                            np.einsum("ij,njk->nik", R_M_A, R_A_i).reshape(-1, 3)], 1)
        b = (p_m - p_A @ R_M_A.T).reshape(-1)
        sol = np.linalg.lstsq(J, b, rcond=None)[0]
        T_M_A = mk(R_M_A, sol[:3])
        X = mk(R_X, sol[3:])
    X_t = np.linalg.inv(T_A_i) @ np.einsum("ij,njk->nik", inv(T_M_A), T_M_m)   # T_i_m(t)
    # per-sample consistency of X
    dev_rot = np.array([rot_deg(X[:3, :3].T @ Xt[:3, :3]) for Xt in X_t])
    dev_tr = np.linalg.norm(X_t[:, :3, 3] - X[:3, 3], axis=1)
    p_imu_M = np.einsum("nij,j->ni", T_M_m @ inv(X), [0, 0, 0, 1])[:, :3]
    resid = np.linalg.norm(apply(T_M_A, T_A_i[:, :3, 3]) - p_imu_M, axis=1)
    return T_M_A, X, {"spread": spread, "dev_rot": dev_rot, "dev_tr": dev_tr,
                      "resid": resid, "p_imu_M": p_imu_M,
                      "p_lio_M": apply(T_M_A, T_A_i[:, :3, 3])}


def parse_axes(spec):
    """'x,z,y' or 'x,-z,y' -> 3x3 S with p_new = S @ p_published."""
    S = np.zeros((3, 3))
    for row, tok in enumerate(spec.replace(" ", "").lower().split(",")):
        sign = -1.0 if tok.startswith("-") else 1.0
        S[row, "xyz".index(tok.lstrip("+-"))] = sign
    if sorted(np.abs(S).sum(0)) != [1, 1, 1] or sorted(np.abs(S).sum(1)) != [1, 1, 1]:
        sys.exit("ERROR: bad --mocap-axes %r (use a permutation like x,z,y)" % spec)
    return S


def _mocap_T(m, S, flip, conj):
    """Mocap poses with positions remapped by S and rotations per convention."""
    T = stack(m)
    T[:, :3, 3] = T[:, :3, 3] @ S.T
    if flip:
        T = _flip_rot(T)
    if conj:
        T[:, :3, :3] = np.einsum("ij,njk,lk->nil", S, T[:, :3, :3], S)
    return T


def ground_truth(ref_odom, ref_moc, tgt_odom, tgt_moc, max_gap=0.1, t_offset=0.0,
                 S=np.eye(3)):
    # reference: match odom to mocap
    ok, m = associate(ref_odom["t"], ref_moc, max_gap, t_offset)
    if ok.sum() < 20:
        raise RuntimeError("Only %d reference odom/mocap matches - check clocks / "
                           "--t-offset" % ok.sum())
    T_A_i = stack(subset(ref_odom, ok))
    m = subset(m, ok)

    # does the axis remap actually fit better? (a mirrored trajectory can't be
    # matched by a rigid transform, so the wrong handedness shows as residual)
    pA = T_A_i[:, :3, 3]
    swap_check = {}
    for name, P in (("as_published", m["p"]), ("remapped", m["p"] @ S.T)):
        T, _ = umeyama(pA, P)
        swap_check[name + "_rmse_m"] = float(np.sqrt(np.mean(
            np.sum((apply(T, pA) - P) ** 2, 1))))

    # quaternion convention: pick the combination that makes X constant
    remap = not np.allclose(S, np.eye(3))
    fits = {}
    for flip in (False, True):
        for conj in ((False, True) if remap else (False,)):
            label = ("world-to-body" if flip else "body-to-world") + \
                    ((", remapped like positions" if conj else ", as published")
                     if remap else "")
            fits[label] = ((flip, conj), fit_reference(T_A_i, _mocap_T(m, S, flip, conj)))
    conv = min(fits, key=lambda c: np.median(fits[c][1][2]["dev_rot"]))
    (flip, conj), (T_M_A, X, q) = fits[conv]
    T_M_m_ref0 = _mocap_T(m, S, flip, conj)[0]

    # target: average poses over the matched (still) window
    ok, m = associate(tgt_odom["t"], tgt_moc, max_gap, t_offset)
    if ok.sum() < 5:
        raise RuntimeError("Only %d target odom/mocap matches - check clocks / "
                           "--t-offset" % ok.sum())
    T_B_i_all = stack(subset(tgt_odom, ok))
    T_M_m_all = _mocap_T(subset(m, ok), S, flip, conj)
    T_B_i = mean_pose(T_B_i_all)
    T_M_mt = mean_pose(T_M_m_all)
    still = {
        "mocap_pos_std_m": float(np.linalg.norm(T_M_m_all[:, :3, 3].std(0))),
        "mocap_rot_std_deg": float(np.sqrt(np.mean(
            [rot_deg(T_M_mt[:3, :3].T @ T[:3, :3]) ** 2 for T in T_M_m_all]))),
        "odom_pos_std_m": float(np.linalg.norm(T_B_i_all[:, :3, 3].std(0))),
        "n": int(ok.sum()),
    }
    T_M_B = T_M_mt @ inv(X) @ inv(T_B_i)
    T_gt = inv(T_M_A) @ T_M_B
    # naive GT: raw mocap start poses compared directly, extrinsic ignored
    T_ref0_M = T_M_m_ref0 @ inv(T_A_i[0])
    T_gt_noX = inv(T_ref0_M) @ T_M_mt @ inv(T_B_i)
    return {"T_gt": T_gt, "T_M_A": T_M_A, "T_M_B": T_M_B, "X": X, "T_B_i": T_B_i,
            "T_gt_noX": T_gt_noX, "ref": q, "still": still,
            "mocap_quat_convention": conv, "axis_check": swap_check,
            "start": {"ref_mocap0": T_M_m_ref0, "tgt_mocap": T_M_mt,
                      "ref_odom0": T_A_i[0], "tgt_odom": T_B_i}}


def _flip_rot(Ts):
    out = Ts.copy()
    out[:, :3, :3] = np.transpose(Ts[:, :3, :3], (0, 2, 1))
    return out


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

def transform_error(T, gt, up, pcd=None):
    """Error of T vs ground truth, expressed in the mocap frame."""
    T_M_A, T_M_B, T_B_i = gt["T_M_A"], gt["T_M_B"], gt["T_B_i"]
    dR = (T_M_A @ T)[:3, :3] @ T_M_B[:3, :3].T
    tilt, yaw = tilt_yaw_deg(dR, up)
    drone_B = T_B_i[:3, 3]
    tr = float(np.linalg.norm(apply(T, drone_B[None]) - apply(gt["T_gt"], drone_B[None])))
    out = {"rotation_deg": rot_deg(dR), "tilt_deg": tilt, "yaw_deg": yaw,
           "translation_at_drone_m": tr}
    if pcd is not None:
        e = np.linalg.norm(apply(T, pcd) - apply(gt["T_gt"], pcd), axis=1)
        out["cloud"] = {"mean_m": float(e.mean()), "median_m": float(np.median(e)),
                        "p95_m": float(np.percentile(e, 95)), "max_m": float(e.max()),
                        "n_points": int(len(e))}
        out["_cloud_err"] = e
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a9a8a2", "#e6e5e0"
C_ALIGNED, C_BASE, C_MOCAP = "#2a78d6", "#a9a8a2", "#0b0b0b"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def make_figures(res, out, args, pcd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) < (3, 5):
        sys.exit("ERROR: matplotlib >= 3.5 needed for the figures "
                 "(pip3 install --user --upgrade 'matplotlib>=3.5')")
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 200,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    al, base, gt = res["aligner"], res["identity"], res["gt"]

    # 1. Error comparison
    panels = [("Rotation error", "deg", al["rotation_deg"], base["rotation_deg"],
               args.rot_thresh, "%.2f°"),
              ("Translation error at drone", "cm", al["translation_at_drone_m"] * 100,
               base["translation_at_drone_m"] * 100, args.trans_thresh * 100, "%.1f cm")]
    if pcd is not None:
        panels.append(("Cloud placement error (95th pct)", "cm", al["cloud"]["p95_m"] * 100,
                       base["cloud"]["p95_m"] * 100, args.trans_thresh * 100, "%.1f cm"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (title, unit, va, vb, thr, fmt) in zip(np.atleast_1d(axes), panels):
        ax.barh([1, 0], [va, vb], color=[C_ALIGNED, C_BASE], height=0.55)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["Aligner", "No alignment\n(identity)"], color=INK2)
        hi = max(va, vb, thr) * 1.3
        ax.set_xlim(0, hi)
        for y, v in ((1, va), (0, vb)):
            ax.text(v + hi * 0.015, y, fmt % v, va="center", color=INK, fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
        ax.axvline(thr, color=INK2, ls=":", lw=1)
        ax.text(thr, 1.45, " pass ≤ " + (fmt % thr), color=INK2, fontsize=8.5, va="bottom")
        ax.set_ylim(-0.5, 1.75)
        ax.set_xlabel(unit)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        _style(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_error_comparison.png"))
    plt.close(fig)

    # 2. Cloud error maps + CDF
    if pcd is not None:
        cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
        ea, eb = al["_cloud_err"] * 100, base["_cloud_err"] * 100
        vmax = max(np.percentile(ea, 99), args.trans_thresh * 100)
        P = apply(gt["T_gt"], pcd)                     # draw in reference frame
        order = np.argsort(P[:, 2])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, e, title in ((axes[0], ea, "Aligner"),
                             (axes[1], eb, "No alignment (identity)")):
            sc = ax.scatter(P[order, 0], P[order, 1], c=np.minimum(e[order], vmax),
                            s=0.6, cmap=cmap, vmin=0, vmax=vmax, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title + "\nmedian %.1f cm · 95th pct %.1f cm"
                         % (np.median(e), np.percentile(e, 95)), loc="left",
                         color=INK, fontsize=10.5)
            ax.set_xlabel("x (m)")
            _style(ax, grid_axis=None)
        axes[0].set_ylabel("y (m)")
        cb = fig.colorbar(sc, ax=axes, shrink=0.85)
        cb.set_label("placement error vs ground truth (cm)%s"
                     % ("  (clipped)" if vmax < max(ea.max(), eb.max()) else ""), color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK2, labelsize=9)
        fig.suptitle("Target cloud: how far each point lands from where it belongs",
                     x=0.01, ha="left", color=INK, fontsize=13)
        fig.savefig(os.path.join(out, "fig_cloud_error.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for e, col, lab in ((eb, C_BASE, "No alignment (identity)"), (ea, C_ALIGNED, "Aligner")):
            x = np.sort(e)
            ax.plot(x, np.linspace(0, 100, len(x)), color=col, lw=2, label=lab)
        ax.axvline(args.trans_thresh * 100, color=INK2, ls=":", lw=1)
        ax.set_xlim(0, max(np.percentile(eb, 99.5), np.percentile(ea, 99.5),
                           args.trans_thresh * 100) * 1.1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("placement error vs ground truth (cm)")
        ax.set_ylabel("% of cloud points at or below")
        ax.set_title("Cloud placement error distribution", loc="left", color=INK)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_cloud_error_cdf.png"))
        plt.close(fig)

    # 3. Ground-truth quality
    q = gt["ref"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    ax.plot(q["p_imu_M"][:, 0], q["p_imu_M"][:, 1], color=C_MOCAP, lw=2, label="Mocap")
    ax.plot(q["p_lio_M"][:, 0], q["p_lio_M"][:, 1], color=C_ALIGNED, lw=1.5, ls="--",
            label="FAST-LIO, fitted")
    d = gt["T_M_B"] @ gt["T_B_i"]
    ax.plot(*d[:2, 3], "o", color=INK, ms=9, label="Target drone (still)")
    hd = d[:3, 0]
    ax.annotate("", xy=d[:2, 3] + 0.4 * hd[:2] / max(np.linalg.norm(hd[:2]), 1e-9),
                xytext=d[:2, 3], arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Reference flight in mocap frame (top-down)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)
    tt = np.arange(len(q["resid"]))
    for ax, y, lab, title in ((axes[1], q["resid"] * 100, "cm", "Reference fit residual"),
                              (axes[2], q["dev_rot"], "deg",
                               "Extrinsic consistency (rotation)")):
        ax.plot(tt, y, color=C_ALIGNED, lw=1.5)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, tt[-1])
        ax.set_xlabel("reference sample")
        ax.set_ylabel(lab)
        ax.set_title("%s · median %.2f %s" % (title, np.median(y), lab), loc="left",
                     color=INK, fontsize=11)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ground_truth_quality.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def start_check(res, up):
    """Raw physical facts, independent of the extrinsic and the axis handling:
    how far apart / how rotated the two start poses were (mocap), what the aligner
    and the GT say, and whether each odometry starts at its own origin."""
    st, gt = res["gt"]["start"], res["gt"]
    a, b = st["ref_mocap0"], st["tgt_mocap"]
    dRw = b[:3, :3] @ a[:3, :3].T                       # rotation between starts, world frame
    _, yaw = tilt_yaw_deg(dRw, up)
    d_drone = lambda T: float(np.linalg.norm(apply(T, gt["T_B_i"][:3, 3][None])[0]
                                             - gt["T_B_i"][:3, 3]))
    return {
        "mocap_start_distance_m": float(np.linalg.norm(b[:3, 3] - a[:3, 3])),
        "mocap_start_rotation_deg": rot_deg(dRw), "mocap_start_yaw_deg": yaw,
        "gt_rotation_deg": rot_deg(gt["T_gt"]), "gt_drone_shift_m": d_drone(gt["T_gt"]),
        "est_rotation_deg": rot_deg(res["T_est"]), "est_drone_shift_m": d_drone(res["T_est"]),
        "ref_odom0_from_origin_m": float(np.linalg.norm(st["ref_odom0"][:3, 3])),
        "ref_odom0_rot_deg": rot_deg(st["ref_odom0"]),
        "tgt_odom_from_origin_m": float(np.linalg.norm(st["tgt_odom"][:3, 3])),
        "tgt_odom_rot_deg": rot_deg(st["tgt_odom"]),
    }


def write_reports(res, out, args, pcd):
    al, base, gt = res["aligner"], res["identity"], res["gt"]
    q, st = gt["ref"], gt["still"]
    rot_ok = al["rotation_deg"] <= args.rot_thresh
    tr_ok = al["translation_at_drone_m"] <= args.trans_thresh
    verdict = "PASS" if rot_ok and tr_ok else "FAIL"
    pf = lambda b: "PASS" if b else "FAIL"
    cm = lambda v: "%.1f cm" % (v * 100)

    warnings = []
    if min(q["spread"][:2]) < 0.3:
        warnings.append("Reference flight is nearly one-dimensional (second principal spread "
                        "%.2f m): its fit to mocap is weakly constrained in rotation."
                        % min(q["spread"][:2]))
    if np.median(q["dev_rot"]) > 2.0:
        warnings.append("Extrinsic estimate is inconsistent across the reference flight "
                        "(median %.1f deg). Check clock offset (--t-offset) and that "
                        "/Odometry is the IMU pose." % np.median(q["dev_rot"]))
    ac = gt["axis_check"]
    if args.mocap_axes.replace(" ", "") != "x,y,z" and \
            ac["as_published_rmse_m"] < 0.7 * ac["remapped_rmse_m"]:
        warnings.append("The mocap positions fit BETTER without the axis remap (%.1f cm vs "
                        "%.1f cm). The y/z swap is probably not real - rerun with "
                        "--mocap-axes x,y,z." % (ac["as_published_rmse_m"] * 100,
                                                 ac["remapped_rmse_m"] * 100))
    up = np.asarray([float(x) for x in args.up.split(",")])
    sc = start_check(res, up / np.linalg.norm(up))
    res["start_check"] = sc
    if sc["mocap_start_rotation_deg"] > 5 * max(sc["est_rotation_deg"], 1.0) or \
            sc["mocap_start_distance_m"] > 0.3 + 5 * sc["est_drone_shift_m"]:
        warnings.append("Mocap says the two starts differ by %.2f m / %.1f deg, but the aligner's "
                        "transform is only %.2f m / %.1f deg. If each cloud were in its own "
                        "start frame these would match - check how the PCDs were built "
                        "(view_raw_clouds.py --odom)."
                        % (sc["mocap_start_distance_m"], sc["mocap_start_rotation_deg"],
                           sc["est_drone_shift_m"], sc["est_rotation_deg"]))
    for nm in ("ref", "tgt"):
        k = "ref_odom0" if nm == "ref" else "tgt_odom"
        if sc[k + "_from_origin_m"] > 0.2 or sc[k + "_rot_deg"] > 5:
            warnings.append("%s odometry does not start at its own origin (%.2f m, %.1f deg): its "
                            "frame is not the drone's start pose."
                            % ("Reference" if nm == "ref" else "Target",
                               sc[k + "_from_origin_m"], sc[k + "_rot_deg"]))
    if st["mocap_pos_std_m"] > 0.02 or st["odom_pos_std_m"] > 0.02:
        warnings.append("Target does not look stationary (mocap std %.1f cm, odom std %.1f cm)."
                        % (st["mocap_pos_std_m"] * 100, st["odom_pos_std_m"] * 100))
    noX = gt["T_gt_noX"]
    ext_effect_rot = rot_deg(gt["T_gt"][:3, :3].T @ noX[:3, :3])
    dB = gt["T_B_i"][:3, 3][None]
    ext_effect_tr = float(np.linalg.norm(apply(gt["T_gt"], dB) - apply(noX, dB)))

    L = []
    L.append("# Alignment validation against mocap ground truth\n")
    L.append("**Verdict: %s** - thresholds set in advance: rotation ≤ %.1f°, translation ≤ %s\n"
             % (verdict, args.rot_thresh, cm(args.trans_thresh)))
    L.append("## Aligner vs ground truth\n")
    L.append("| Metric | Aligner | No alignment (identity) | Threshold | Result |")
    L.append("|---|---|---|---|---|")
    L.append("| Rotation error | %.2f° | %.2f° | %.1f° | %s |"
             % (al["rotation_deg"], base["rotation_deg"], args.rot_thresh, pf(rot_ok)))
    L.append("| &nbsp;&nbsp;tilt (roll/pitch) | %.2f° | %.2f° | | |" % (al["tilt_deg"], base["tilt_deg"]))
    L.append("| &nbsp;&nbsp;yaw | %+.2f° | %+.2f° | | |" % (al["yaw_deg"], base["yaw_deg"]))
    L.append("| Translation error at drone | %s | %s | %s | %s |"
             % (cm(al["translation_at_drone_m"]), cm(base["translation_at_drone_m"]),
                cm(args.trans_thresh), pf(tr_ok)))
    if pcd is not None:
        a, b = al["cloud"], base["cloud"]
        L.append("")
        L.append("## Target cloud placement error (%d points)\n" % a["n_points"])
        L.append("How far each target point lands from where the ground truth puts it.\n")
        L.append("| | Mean | Median | 95th pct | Max |\n|---|---|---|---|---|")
        L.append("| Aligner | %s | %s | %s | %s |" % (cm(a["mean_m"]), cm(a["median_m"]),
                                                   cm(a["p95_m"]), cm(a["max_m"])))
        L.append("| No alignment (identity) | %s | %s | %s | %s |"
                 % (cm(b["mean_m"]), cm(b["median_m"]), cm(b["p95_m"]), cm(b["max_m"])))
    L.append("")
    L.append("## Sanity check: how different were the two starts?\n")
    L.append("Straight from mocap - unaffected by the extrinsic or axis handling. If each cloud "
             "is in its own FAST-LIO start frame, the ground truth and a good aligner should "
             "both be about this size.\n")
    L.append("| | Shift of drone position | Rotation |\n|---|---|---|")
    L.append("| Physical start change (mocap) | %s | %.1f° (yaw %+.1f°) |"
             % (cm(sc["mocap_start_distance_m"]), sc["mocap_start_rotation_deg"],
                sc["mocap_start_yaw_deg"]))
    L.append("| Ground truth | %s | %.1f° |" % (cm(sc["gt_drone_shift_m"]), sc["gt_rotation_deg"]))
    L.append("| Aligner (T_est) | %s | %.1f° |" % (cm(sc["est_drone_shift_m"]), sc["est_rotation_deg"]))
    L.append("| Reference odometry first pose | %s from origin | %.1f° from identity |"
             % (cm(sc["ref_odom0_from_origin_m"]), sc["ref_odom0_rot_deg"]))
    L.append("| Target odometry (mean pose) | %s from origin | %.1f° from identity |"
             % (cm(sc["tgt_odom_from_origin_m"]), sc["tgt_odom_rot_deg"]))
    L.append("")
    L.append("## Ground-truth quality\n")
    L.append("| Check | Value |\n|---|---|")
    L.append("| Reference fit residual (RMSE / max) | %s / %s over %d poses |"
             % (cm(np.sqrt(np.mean(q["resid"] ** 2))), cm(q["resid"].max()), len(q["resid"])))
    L.append("| Reference trajectory spread (principal axes) | %s m |"
             % ", ".join("%.2f" % s for s in q["spread"]))
    L.append("| Extrinsic consistency across reference (median / 95th pct) | %.2f° / %.2f°, "
             "%s / %s |" % (np.median(q["dev_rot"]), np.percentile(q["dev_rot"], 95),
                            cm(np.median(q["dev_tr"])), cm(np.percentile(q["dev_tr"], 95))))
    L.append("| Target stillness (mocap pos std / rot std, odom pos std) | %s / %.2f°, %s over %d samples |"
             % (cm(st["mocap_pos_std_m"]), st["mocap_rot_std_deg"], cm(st["odom_pos_std_m"]), st["n"]))
    L.append("| Naive GT (raw mocap start poses, extrinsic ignored) differs by | %.2f°, %s |"
             % (ext_effect_rot, cm(ext_effect_tr)))
    L.append("| Mocap quaternion convention (auto-detected) | %s |" % gt["mocap_quat_convention"])
    ac = gt["axis_check"]
    L.append("| Mocap axis remap `%s`: reference fit RMSE as published / remapped | %s / %s |"
             % (args.mocap_axes, cm(ac["as_published_rmse_m"]), cm(ac["remapped_rmse_m"])))
    if warnings:
        L.append("")
        for w in warnings:
            L.append("> **Warning:** " + w + "\n")
    np.set_printoptions(precision=5, suppress=True)
    X = gt["X"]
    L.append("\n## Transforms\n")
    L.append("T_est (aligner, target LIO → reference LIO):\n```\n%s\n```" % res["T_est"])
    L.append("T_gt (mocap-derived, target LIO → reference LIO):\n```\n%s\n```" % gt["T_gt"])
    L.append("Estimated marker→IMU extrinsic: rotation %.2f° (rpy %s°), lever arm %s\n"
             % (rot_deg(X), np.round(Rot.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True), 2),
                cm(np.linalg.norm(X[:3, 3]))))
    L.append("## Method\n")
    L.append("1. Reference flight (moving): fit its FAST-LIO trajectory to mocap, and estimate the "
             "marker→IMU extrinsic from the per-timestamp pose difference.\n"
             "2. Target (stationary): average its mocap pose; FAST-LIO's frame is the IMU pose "
             "at start-up, so this fixes the target frame in mocap.\n"
             "3. Ground truth = relative pose between the two frames. The aligner's transform and "
             "the identity (do-nothing) baseline are compared against it.\n")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L))

    clean = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    js = {"verdict": verdict,
          "thresholds": {"rotation_deg": args.rot_thresh, "translation_m": args.trans_thresh},
          "aligner": clean(al), "identity_baseline": clean(base),
          "T_est": res["T_est"].tolist(), "T_gt": gt["T_gt"].tolist(),
          "T_gt_without_extrinsic": noX.tolist(),
          "extrinsic_T_imu_marker": X.tolist(),
          "ground_truth_quality": {
              "reference_fit_rmse_m": float(np.sqrt(np.mean(q["resid"] ** 2))),
              "reference_fit_max_m": float(q["resid"].max()),
              "reference_spread_m": q["spread"].tolist(),
              "extrinsic_rot_dev_median_deg": float(np.median(q["dev_rot"])),
              "extrinsic_rot_dev_p95_deg": float(np.percentile(q["dev_rot"], 95)),
              "extrinsic_trans_dev_median_m": float(np.median(q["dev_tr"])),
              "extrinsic_effect_rot_deg": ext_effect_rot,
              "extrinsic_effect_trans_m": ext_effect_tr,
              "target_stillness": st,
              "mocap_quat_convention": gt["mocap_quat_convention"],
              "mocap_axes": args.mocap_axes, "axis_check": gt["axis_check"]},
          "start_check": sc, "warnings": warnings}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(js, f, indent=2)
    return verdict, warnings


# --------------------------------------------------------------------------- #

def run(ref_odom, ref_moc, tgt_odom, tgt_moc, T_est, args, pcd=None):
    up = np.asarray([float(x) for x in args.up.split(",")])
    up /= np.linalg.norm(up)
    gt = ground_truth(ref_odom, ref_moc, tgt_odom, tgt_moc, args.max_gap, args.t_offset,
                      parse_axes(args.mocap_axes))
    res = {"T_est": T_est, "gt": gt,
           "aligner": transform_error(T_est, gt, up, pcd),
           "identity": transform_error(np.eye(4), gt, up, pcd)}
    os.makedirs(args.out, exist_ok=True)
    make_figures(res, args.out, args, pcd)
    verdict, warnings = write_reports(res, args.out, args, pcd)
    return res, verdict, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-bag", required=True, help="reference bag (drone moving)")
    ap.add_argument("--tgt-bag", required=True, help="target bag (drone stationary)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--T", help="T_est file: .npy, or text with 16 numbers")
    g.add_argument("--T-bag", help="bag containing the node's published matrix")
    ap.add_argument("--T-topic", default="/target_to_reference_matrix")
    ap.add_argument("--tgt-pcd", help="target cloud (frame B) for placement error")
    ap.add_argument("--odom-topic", default="/Odometry")
    ap.add_argument("--mocap-topic", default="/mocap_output113")
    ap.add_argument("--max-gap", type=float, default=0.1, help="max mocap gap to interpolate (s)")
    ap.add_argument("--t-offset", type=float, default=0.0,
                    help="added to odom stamps before matching mocap (s)")
    ap.add_argument("--mocap-axes", default="x,z,y",
                    help="remap published mocap axes: new (x,y,z) = these published "
                         "components. Default x,z,y swaps y and z. Use x,y,z for none, "
                         "or e.g. x,-z,y for a proper Y-up -> Z-up rotation.")
    ap.add_argument("--up", default="0,0,1", help="up axis AFTER the remap")
    ap.add_argument("--rot-thresh", type=float, default=1.0, help="deg")
    ap.add_argument("--trans-thresh", type=float, default=0.10, help="m")
    ap.add_argument("--out", default="alignment_eval")
    args = ap.parse_args(argv)

    T_est = load_T(args)
    print("Loading bags...")
    data = {}
    for key, bag, topic, skip in (("ref odom", args.ref_bag, args.odom_topic, False),
                                  ("ref mocap", args.ref_bag, args.mocap_topic, True),
                                  ("tgt odom", args.tgt_bag, args.odom_topic, False),
                                  ("tgt mocap", args.tgt_bag, args.mocap_topic, True)):
        data[key] = load_poses(bag, topic, skip)
        t = data[key]["t"]
        print("  %-10s %6d msgs  %7.1f s long  starts at %.3f"
              % (key, len(t), t[-1] - t[0], t[0]))
    for name in ("ref", "tgt"):
        to, tm = data[name + " odom"]["t"], data[name + " mocap"]["t"]
        overlap = min(to[-1] + args.t_offset, tm[-1]) - max(to[0] + args.t_offset, tm[0])
        if overlap <= 0:
            sys.exit("ERROR: %s bag: odometry and mocap time ranges don't overlap (start "
                     "difference %.3f s). Clocks differ - set --t-offset to about "
                     "(mocap start - odom start)." % (name, tm[0] - to[0]))
    pcd = load_pcd_xyz(args.tgt_pcd) if args.tgt_pcd else None

    res, verdict, warnings = run(data["ref odom"], data["ref mocap"], data["tgt odom"],
                                 data["tgt mocap"], T_est, args, pcd)
    al, base = res["aligner"], res["identity"]
    print("\n=== %s ===" % verdict)
    print("                 aligner      identity baseline")
    print("rotation        %6.2f deg   %6.2f deg" % (al["rotation_deg"], base["rotation_deg"]))
    print("  tilt / yaw    %5.2f / %+5.2f  %5.2f / %+5.2f"
          % (al["tilt_deg"], al["yaw_deg"], base["tilt_deg"], base["yaw_deg"]))
    print("translation     %6.1f cm    %6.1f cm" % (al["translation_at_drone_m"] * 100,
                                                    base["translation_at_drone_m"] * 100))
    if pcd is not None:
        print("cloud p95       %6.1f cm    %6.1f cm" % (al["cloud"]["p95_m"] * 100,
                                                        base["cloud"]["p95_m"] * 100))
    q = res["gt"]["ref"]
    print("GT: ref fit RMSE %.1f cm, extrinsic consistency %.2f deg median, mocap quats %s"
          % (np.sqrt(np.mean(q["resid"] ** 2)) * 100, np.median(q["dev_rot"]),
             res["gt"]["mocap_quat_convention"]))
    sc = res["start_check"]
    print("Start poses: mocap %.2f m / %.1f deg apart | GT %.2f m / %.1f deg | aligner %.2f m / %.1f deg"
          % (sc["mocap_start_distance_m"], sc["mocap_start_rotation_deg"], sc["gt_drone_shift_m"],
             sc["gt_rotation_deg"], sc["est_drone_shift_m"], sc["est_rotation_deg"]))
    ac = res["gt"]["axis_check"]
    print("Mocap axes %s: ref fit RMSE %.1f cm as published, %.1f cm remapped"
          % (args.mocap_axes, ac["as_published_rmse_m"] * 100, ac["remapped_rmse_m"] * 100))
    for w in warnings:
        print("WARNING:", w)
    print("Outputs in %s/" % args.out)


if __name__ == "__main__":
    main()
