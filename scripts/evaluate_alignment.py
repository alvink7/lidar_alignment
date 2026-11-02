#!/usr/bin/env python3
"""
evaluate_alignment.py - prove cloud_aligner's transform against mocap ground truth.

Method
------
  A = reference bag's FAST-LIO frame (camera_init of the run that made reference PCD)
  B = target bag's FAST-LIO frame
  M = mocap world frame (shared by both flights)

  1. Reference bag:  fit T_M_A  (Umeyama, rigid, no scale) from its /Odometry vs mocap.
                     This is the ONLY fit that touches the evaluation and it does not
                     involve the aligner.
  2. Target bag:     p_M = T_M_A @ T_est @ p_B(t), compared point-by-point to the target
                     bag's mocap. NO re-alignment - any error in T_est shows up here.
  3. Oracle:         target /Odometry fitted straight to its own mocap. This is the floor
                     (LIO drift + mocap/IMU lever arm + timing), i.e. the best ANY single
                     rigid transform could do.

  Alignment error added = step 2 vs step 3.
  Transform error       = T_M_A @ T_est   vs   T_M_B (oracle fit), reported as
                          rotation (total / tilt / yaw) and translation at the target
                          trajectory centroid.

Outputs (in --out)
------------------
  summary.md              presentation-ready tables + PASS/FAIL
  summary.json            every number, machine-readable
  per_pose_errors.csv     t, error (aligned), error (oracle), positions
  fig_trajectories.png    top-down + side view: mocap truth vs aligned vs oracle
  fig_error_over_time.png per-pose position error, aligned vs oracle floor
  fig_error_cdf.png       cumulative distribution of position error

Usage
-----
  # transform from a file (4x4 as .npy, or 16 numbers in a text file, row-major)
  python3 evaluate_alignment.py --ref-bag ref.bag --tgt-bag tgt.bag --T T_est.txt

  # transform read from a bag where /target_to_reference_matrix was recorded
  python3 evaluate_alignment.py --ref-bag ref.bag --tgt-bag tgt_run.bag \
      --T-bag tgt_run.bag

IMPORTANT: T_est only means something relative to the exact FAST-LIO runs it connects.
  - The reference /Odometry must come from the same FAST-LIO run that produced the
    reference PCD (same camera_init).
  - The target /Odometry must come from the same FAST-LIO run the node aligned
    (record /Odometry and /target_to_reference_matrix together during the live run).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Bag loading (rosbag only imported when actually reading bags)
# --------------------------------------------------------------------------- #

def _stamp(msg, t_bag):
    h = getattr(msg, "header", None)
    if h is not None and hasattr(h, "stamp"):
        s = h.stamp.to_sec()
        if s > 0:
            return s
    return t_bag.to_sec()


def _get_path(obj, path):
    for p in path.split("."):
        if not hasattr(obj, p):
            return None
        obj = getattr(obj, p)
    return obj


def _position(msg):
    """Pull an xyz position out of the common message layouts."""
    if all(hasattr(msg, f) for f in ("posx", "posy", "posz")):      # hiperlab
        return [msg.posx, msg.posy, msg.posz]
    for path in ("pose.pose.position", "pose.position", "transform.translation",
                 "position", "point"):
        o = _get_path(msg, path)
        if o is not None and all(hasattr(o, a) for a in ("x", "y", "z")):
            return [o.x, o.y, o.z]
    raise ValueError("Can't find a position in message type %s. Fields: %s"
                     % (msg._type, msg.__slots__))


def load_positions(bag_path, topic, skip_zero=True):
    import rosbag
    t, p = [], []
    with rosbag.Bag(bag_path) as bag:
        for _, msg, tb in bag.read_messages(topics=[topic]):
            xyz = _position(msg)
            if not np.all(np.isfinite(xyz)):
                continue
            if skip_zero and xyz[0] == 0.0 and xyz[1] == 0.0 and xyz[2] == 0.0:
                continue                          # mocap dropout / inactive
            t.append(_stamp(msg, tb))
            p.append(xyz)
    if not t:
        sys.exit("ERROR: no usable messages on %s in %s" % (topic, bag_path))
    t, p = np.asarray(t), np.asarray(p, dtype=float)
    order = np.argsort(t)
    return t[order], p[order]


def load_T(args):
    if args.T:
        if args.T.endswith(".npy"):
            T = np.load(args.T)
        else:
            with open(args.T) as f:
                nums = [float(x) for x in f.read().replace(",", " ").replace("[", " ")
                        .replace("]", " ").split()]
            T = np.asarray(nums)
        T = np.asarray(T, dtype=float).reshape(4, 4)
    else:
        import rosbag
        T = None
        with rosbag.Bag(args.T_bag) as bag:
            for _, msg, _ in bag.read_messages(topics=[args.T_topic]):
                T = np.asarray(msg.data, dtype=float).reshape(4, 4)   # keep last
        if T is None:
            sys.exit("ERROR: no %s message in %s" % (args.T_topic, args.T_bag))
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        sys.exit("ERROR: T_est bottom row is not [0 0 0 1]:\n%s" % T)
    R = T[:3, :3]
    if abs(np.linalg.det(R) - 1) > 1e-3 or not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        sys.exit("ERROR: T_est rotation block is not a proper rotation:\n%s" % T)
    return T


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #

def umeyama(src, dst):
    """Rigid (no scale) fit: dst ~= R @ src + t. Returns 4x4 T and singular values."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, mu_d - R @ mu_s
    # excitation check: spread of the trajectory along its principal axes (m)
    spread = np.sqrt(np.linalg.svd(S, compute_uv=False) ** 2 / len(src))
    return T, spread


def apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def associate(t_query, t_ref, p_ref, max_gap, t_offset=0.0):
    """Linearly interpolate p_ref at t_query (+offset). Returns (mask, positions)."""
    tq = t_query + t_offset
    i = np.searchsorted(t_ref, tq)
    ok = (i > 0) & (i < len(t_ref))
    i = np.clip(i, 1, len(t_ref) - 1)
    t0, t1 = t_ref[i - 1], t_ref[i]
    ok &= (t1 - t0) <= max_gap
    w = np.where(t1 > t0, (tq - t0) / np.maximum(t1 - t0, 1e-12), 0.0)[:, None]
    p = (1 - w) * p_ref[i - 1] + w * p_ref[i]
    return ok, p


def stats(e):
    return {"rmse_m": float(np.sqrt(np.mean(e ** 2))), "mean_m": float(np.mean(e)),
            "median_m": float(np.median(e)), "std_m": float(np.std(e)),
            "max_m": float(np.max(e)), "n": int(len(e))}


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rot_split_deg(R, up):
    """Split a rotation error into tilt (moves the up axis) and yaw (about up)."""
    tilt = float(np.degrees(np.arccos(np.clip((R @ up) @ up, -1, 1))))
    # yaw: rotate a horizontal axis, project back onto the horizontal plane
    h = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0])
    h /= np.linalg.norm(h)
    hr = R @ h
    hr -= (hr @ up) * up
    yaw = float(np.degrees(np.arctan2(np.cross(h, hr) @ up, h @ hr)))
    return tilt, yaw


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(ref_odom, ref_moc, tgt_odom, tgt_moc, T_est, max_gap=0.1,
             t_offset=0.0, up=(0, 0, 1)):
    up = np.asarray(up, float) / np.linalg.norm(up)

    # --- Step 1: reference LIO frame A -> mocap M
    ok, m = associate(ref_odom[0], ref_moc[0], ref_moc[1], max_gap, t_offset)
    if ok.sum() < 10:
        raise RuntimeError("Only %d reference odom/mocap matches - check timestamps "
                           "or --t-offset" % ok.sum())
    pA, pM = ref_odom[1][ok], m[ok]
    T_M_A, ref_spread = umeyama(pA, pM)
    ref_resid = np.linalg.norm(apply(T_M_A, pA) - pM, axis=1)

    # --- Step 2: target through the aligner, no re-fit
    ok, m = associate(tgt_odom[0], tgt_moc[0], tgt_moc[1], max_gap, t_offset)
    if ok.sum() < 10:
        raise RuntimeError("Only %d target odom/mocap matches - check timestamps "
                           "or --t-offset" % ok.sum())
    tB, pB, gt = tgt_odom[0][ok], tgt_odom[1][ok], m[ok]
    T_M_B_est = T_M_A @ T_est
    p_aligned = apply(T_M_B_est, pB)
    e_aligned = np.linalg.norm(p_aligned - gt, axis=1)

    # --- Step 3: oracle - best possible rigid placement of the target run
    T_M_B_gt, tgt_spread = umeyama(pB, gt)
    p_oracle = apply(T_M_B_gt, pB)
    e_oracle = np.linalg.norm(p_oracle - gt, axis=1)

    # --- Transform error (expressed in mocap frame)
    dR = T_M_B_est[:3, :3] @ T_M_B_gt[:3, :3].T
    tilt, yaw = rot_split_deg(dR, up)
    c = pB.mean(0)                                         # target centroid, frame B
    trans_at_centroid = float(np.linalg.norm(apply(T_M_B_est, c[None]) -
                                             apply(T_M_B_gt, c[None])))
    T_A_B_gt = np.linalg.inv(T_M_A) @ T_M_B_gt

    s_al, s_or = stats(e_aligned), stats(e_oracle)
    return {
        "T_est": T_est, "T_A_B_gt": T_A_B_gt, "T_M_A": T_M_A,
        "transform_error": {
            "rotation_deg": rot_angle_deg(dR), "tilt_deg": tilt, "yaw_deg": yaw,
            "translation_at_target_centroid_m": trans_at_centroid,
        },
        "ate_aligned": s_al, "ate_oracle": s_or,
        "ate_reference_fit": stats(ref_resid),
        "added_rmse_m": s_al["rmse_m"] - s_or["rmse_m"],
        "excitation_m": {"reference": ref_spread.tolist(), "target": tgt_spread.tolist()},
        "series": {"t": tB - tB[0], "gt": gt, "aligned": p_aligned, "oracle": p_oracle,
                   "e_aligned": e_aligned, "e_oracle": e_oracle,
                   "ref_gt": pM},
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a9a8a2", "#e6e5e0"
C_ALIGNED, C_ORACLE = "#2a78d6", "#eb6834"      # categorical slots 1, 2


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def make_figures(r, out, trans_thresh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.dpi": 200})
    s = r["series"]

    # Trajectories: top-down and side view
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, (i, j, xl, yl, title) in zip(axes, [(0, 1, "x (m)", "y (m)", "Top-down"),
                                                (0, 2, "x (m)", "z (m)", "Side view")]):
        ax.plot(s["ref_gt"][:, i], s["ref_gt"][:, j], color=GRID, lw=3,
                label="Reference flight (mocap)", zorder=1)
        ax.plot(s["gt"][:, i], s["gt"][:, j], color=INK, lw=2,
                label="Target flight, mocap truth", zorder=3)
        ax.plot(s["oracle"][:, i], s["oracle"][:, j], color=C_ORACLE, lw=2, ls="--",
                label="Target LIO, best-possible fit (oracle)", zorder=2)
        ax.plot(s["aligned"][:, i], s["aligned"][:, j], color=C_ALIGNED, lw=2,
                label="Target LIO via aligner transform", zorder=4)
        ax.plot(*s["gt"][0, [i, j]], "o", color=INK, ms=8, zorder=5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        _style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.suptitle("Target flight placed in the mocap frame by the aligner's transform",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "fig_trajectories.png"))
    plt.close(fig)

    # Error over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s["t"], s["e_oracle"] * 100, color=C_ORACLE, lw=2,
            label="Oracle floor (LIO drift only)")
    ax.plot(s["t"], s["e_aligned"] * 100, color=C_ALIGNED, lw=2,
            label="Via aligner transform")
    ax.axhline(trans_thresh * 100, color=INK2, lw=1, ls=":")
    ax.text(s["t"][-1], trans_thresh * 100, " threshold %.0f cm" % (trans_thresh * 100),
            va="center", ha="left", color=INK2, fontsize=9, clip_on=False)
    ax.set_xlabel("time since start of target flight (s)")
    ax.set_xlim(s["t"][0], s["t"][-1])
    ax.set_ylabel("position error vs mocap (cm)")
    ax.set_ylim(0, max(trans_thresh, s["e_aligned"].max(), s["e_oracle"].max()) * 100 * 1.15)
    ax.set_title("Position error over the target flight", loc="left", color=INK, pad=26)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.0), borderaxespad=0)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_error_over_time.png"))
    plt.close(fig)

    # CDF
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for e, col, lab in ((s["e_oracle"], C_ORACLE, "Oracle floor"),
                        (s["e_aligned"], C_ALIGNED, "Via aligner transform")):
        x = np.sort(e) * 100
        ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, color=col, lw=2, label=lab)
    ax.axvline(trans_thresh * 100, color=INK2, lw=1, ls=":")
    ax.set_xlabel("position error vs mocap (cm)")
    ax.set_ylabel("% of poses at or below")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.set_title("Error distribution", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_error_cdf.png"))
    plt.close(fig)


def write_reports(r, out, args):
    te = r["transform_error"]
    al, orc, ref = r["ate_aligned"], r["ate_oracle"], r["ate_reference_fit"]
    rot_ok = te["rotation_deg"] <= args.rot_thresh
    tr_ok = te["translation_at_target_centroid_m"] <= args.trans_thresh
    add_ok = r["added_rmse_m"] <= args.trans_thresh
    verdict = "PASS" if (rot_ok and tr_ok and add_ok) else "FAIL"
    pf = lambda b: "PASS" if b else "FAIL"
    cm = lambda v: "%.1f cm" % (v * 100)

    # excitation warning: weakest axis of trajectory spread
    weak = min(min(r["excitation_m"]["reference"][:2]), min(r["excitation_m"]["target"][:2]))
    exc_note = ("" if weak > 0.3 else
                "\n> **Warning:** a trajectory is nearly one-dimensional (second principal "
                "spread %.2f m). The Umeyama fit poorly constrains rotation about that "
                "line, so the ground truth itself is weak there.\n" % weak)

    md = []
    md.append("# Alignment validation against mocap ground truth\n")
    md.append("**Verdict: %s** (thresholds set in advance: rotation <= %.1f deg, "
              "translation <= %s)\n" % (verdict, args.rot_thresh, cm(args.trans_thresh)))
    md.append("## Transform error (aligner vs mocap-derived ground truth)\n")
    md.append("| Metric | Value | Threshold | Result |\n|---|---|---|---|")
    md.append("| Rotation error (total) | %.2f deg | %.1f deg | %s |"
              % (te["rotation_deg"], args.rot_thresh, pf(rot_ok)))
    md.append("| &nbsp;&nbsp;tilt (roll/pitch) | %.2f deg | | |" % te["tilt_deg"])
    md.append("| &nbsp;&nbsp;yaw | %+.2f deg | | |" % te["yaw_deg"])
    md.append("| Translation error at target centroid | %s | %s | %s |"
              % (cm(te["translation_at_target_centroid_m"]), cm(args.trans_thresh), pf(tr_ok)))
    md.append("")
    md.append("## Trajectory error (target flight vs mocap)\n")
    md.append("| | RMSE | Mean | Median | Max | Poses |\n|---|---|---|---|---|---|")
    md.append("| Via aligner transform | %s | %s | %s | %s | %d |"
              % (cm(al["rmse_m"]), cm(al["mean_m"]), cm(al["median_m"]), cm(al["max_m"]), al["n"]))
    md.append("| Oracle floor (best rigid fit) | %s | %s | %s | %s | %d |"
              % (cm(orc["rmse_m"]), cm(orc["mean_m"]), cm(orc["median_m"]), cm(orc["max_m"]), orc["n"]))
    md.append("| **Error added by alignment** | **%s** | | | | |" % cm(r["added_rmse_m"]))
    md.append("")
    md.append("## Ground-truth noise floor\n")
    md.append("Reference flight LIO-to-mocap fit residual: RMSE %s, max %s over %d poses. "
              "Errors at or below this level are within measurement precision.\n"
              % (cm(ref["rmse_m"]), cm(ref["max_m"]), ref["n"]))
    md.append(exc_note)
    md.append("## Transforms\n")
    np.set_printoptions(precision=4, suppress=True)
    md.append("T_est (aligner, target LIO -> reference LIO):\n```\n%s\n```" % r["T_est"])
    md.append("T_gt (mocap-derived, target LIO -> reference LIO):\n```\n%s\n```" % r["T_A_B_gt"])
    md.append("\n## Method\n")
    md.append("1. Fit the reference flight's LIO frame to mocap (rigid, one fit; "
              "independent of the aligner).\n"
              "2. Map the target flight's LIO positions through the aligner's transform and "
              "that fit into mocap, and compare to the target's mocap with **no "
              "re-alignment**.\n"
              "3. Oracle: fit the target LIO straight to its own mocap - the best any rigid "
              "transform could do. The gap between 2 and 3 is the alignment's own error.\n")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md))

    js = {k: v for k, v in r.items() if k != "series"}
    for k in ("T_est", "T_A_B_gt", "T_M_A"):
        js[k] = np.asarray(js[k]).tolist()
    js["thresholds"] = {"rotation_deg": args.rot_thresh, "translation_m": args.trans_thresh}
    js["verdict"] = verdict
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(js, f, indent=2)

    s = r["series"]
    with open(os.path.join(out, "per_pose_errors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "err_aligned_m", "err_oracle_m", "gt_x", "gt_y", "gt_z",
                    "aligned_x", "aligned_y", "aligned_z"])
        for k in range(len(s["t"])):
            w.writerow(["%.4f" % s["t"][k], "%.5f" % s["e_aligned"][k],
                        "%.5f" % s["e_oracle"][k]] +
                       ["%.5f" % v for v in s["gt"][k]] +
                       ["%.5f" % v for v in s["aligned"][k]])
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-bag", required=True)
    ap.add_argument("--tgt-bag", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--T", help="4x4 T_est file (.npy, or text with 16 numbers row-major)")
    g.add_argument("--T-bag", help="bag containing the node's published matrix")
    ap.add_argument("--T-topic", default="/target_to_reference_matrix")
    ap.add_argument("--odom-topic", default="/Odometry")
    ap.add_argument("--mocap-topic", default="/mocap_output113")
    ap.add_argument("--max-gap", type=float, default=0.1,
                    help="max mocap sample gap to interpolate across (s)")
    ap.add_argument("--t-offset", type=float, default=0.0,
                    help="added to odom stamps before matching mocap (s)")
    ap.add_argument("--up", default="0,0,1", help="mocap up axis, e.g. 0,1,0 for Y-up")
    ap.add_argument("--rot-thresh", type=float, default=1.0, help="deg")
    ap.add_argument("--trans-thresh", type=float, default=0.10, help="m")
    ap.add_argument("--out", default="alignment_eval")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    T_est = load_T(args)
    print("Loading bags...")
    ref_odom = load_positions(args.ref_bag, args.odom_topic, skip_zero=False)
    ref_moc = load_positions(args.ref_bag, args.mocap_topic)
    tgt_odom = load_positions(args.tgt_bag, args.odom_topic, skip_zero=False)
    tgt_moc = load_positions(args.tgt_bag, args.mocap_topic)
    for name, (t, _) in (("ref odom", ref_odom), ("ref mocap", ref_moc),
                         ("tgt odom", tgt_odom), ("tgt mocap", tgt_moc)):
        print("  %-10s %6d msgs  %7.1f s long  starts at %.3f" % (name, len(t), t[-1] - t[0], t[0]))
    for bag, (to, _), (tm, _) in (("reference", ref_odom, ref_moc), ("target", tgt_odom, tgt_moc)):
        overlap = min(to[-1], tm[-1] - args.t_offset) - max(to[0], tm[0] - args.t_offset)
        if overlap <= 0:
            sys.exit("ERROR: %s bag: odometry and mocap time ranges don't overlap "
                     "(start difference %.3f s). Clocks differ - set --t-offset to "
                     "about (mocap start - odom start)." % (bag, tm[0] - to[0]))

    r = evaluate(ref_odom, ref_moc, tgt_odom, tgt_moc, T_est, args.max_gap,
                 args.t_offset, [float(x) for x in args.up.split(",")])
    make_figures(r, args.out, args.trans_thresh)
    verdict = write_reports(r, args.out, args)

    te = r["transform_error"]
    print("\n=== %s ===" % verdict)
    print("rotation error   %.2f deg (tilt %.2f, yaw %+.2f)"
          % (te["rotation_deg"], te["tilt_deg"], te["yaw_deg"]))
    print("translation err  %.1f cm at target centroid"
          % (te["translation_at_target_centroid_m"] * 100))
    print("ATE aligned      %.1f cm RMSE   oracle %.1f cm   added %.1f cm"
          % (r["ate_aligned"]["rmse_m"] * 100, r["ate_oracle"]["rmse_m"] * 100,
             r["added_rmse_m"] * 100))
    print("GT noise floor   %.1f cm RMSE (reference fit)"
          % (r["ate_reference_fit"]["rmse_m"] * 100))
    print("Outputs in %s/" % args.out)


if __name__ == "__main__":
    main()
