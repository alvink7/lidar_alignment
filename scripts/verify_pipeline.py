#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_pipeline.py

Offline, pure-library run of the full pipeline -- Scan Context retrieval
(scancontext/python) -> align_target refinement (alignment_core.py) --
against a revisit bag, with no ROS node and no live topics involved. Feeds
each query observation through KeyframeDB.lookup() (Seam A) then
align_target() on the CONSERVATIVE path (skip_teaser=False, Seam B's fast
path deliberately NOT used here -- see run_verification()), and reports
where each query landed: no match, failed the coverage/RMSE quality gate,
disagreed with TEASER on yaw, or was accepted.

Six seams are isolated below (load_reference, iterate_queries,
build_sc_query_cloud, matcher_query, load_align_params, run_align) so a
different DB format or bag layout only requires editing those functions;
run_verification()'s gate order and findings.md's structure are fixed.

Conservative path, on purpose: skip_teaser=False makes align_target run
TEASER independently of the Scan Context yaw, so the yaw-agreement gate is
a real cross-check between two separately-derived yaw estimates, not a
seeded value agreeing with itself.
"""
import argparse
import math
import os
import sys
from collections import Counter

import numpy as np
import open3d as o3d
import rosbag

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "..", "scancontext", "python"))
import sc_keyframes as sck
from sc_relocalize import KeyframeDB, make_T_yaw
from alignment_core import align_target

import faiss


# =============================================================================
# small helpers
# =============================================================================
def wrap_deg(d):
    """Wrap an angle in degrees to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def yaw_from_T(T):
    """Planar yaw (deg) of a 4x4, same CCW-about-+Z convention as
    alignment_core.constrain_planar / sc_keyframes.pose_xy_yaw."""
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def nearest_kf_by_pose(db, ref_pose):
    """(nearest_index, distance_m): the DB keyframe whose pose is closest in
    XY to ref_pose, regardless of what Scan Context itself retrieved. Splits
    "no keyframe was ever near here" (correct out-of-map no_match) from "a
    keyframe was right there and retrieval still missed" (a real bug) --
    see db[i] -> (PreparedReference, pose), pose a 4x4 body->world."""
    q = np.asarray(ref_pose)[:2, 3]
    best_i, best_d = -1, float("inf")
    for i in range(len(db)):
        _prepared_ref, kf_pose = db[i]
        p = np.asarray(kf_pose)[:2, 3]
        d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def make_faiss_resources():
    """One persistent GPU resource handle, reused across every align_target
    call (never per-call) -- falls back to CPU (None) with no GPU / build."""
    try:
        return faiss.StandardGpuResources()
    except Exception:
        return None


# =============================================================================
# Seam 1 -- load reference
# =============================================================================
def load_reference(ref_dir):
    """(matcher, db): matcher.query(pts) -> (index, yaw_rad); db[index] ->
    (PreparedReference, pose). KeyframeDB already owns both -- it loads
    manifest.json + sc_params.json + the keyframe .pcds and replays them
    through a ScanContextMatcher (see sc_relocalize.KeyframeDB)."""
    db = KeyframeDB(ref_dir)
    return db.matcher, db


# =============================================================================
# Seam 2 -- iterate queries
# =============================================================================
def iterate_queries(bag_path, cloud_topic, odom_topic, query_every,
                     tgt_accumulate_sec):
    """Yields (window, ref_pose, dense_target_cloud, gt_pose) per query.

    window: time-ordered SyncedScan list up to and including the query
        instant (feeds build_sc_query_cloud). ref_pose: the query instant's
        body->world pose. dense_target_cloud: an o3d cloud accumulated over
        tgt_accumulate_sec -- deliberately a LONGER, coarser-voxel window
        than the ~0.3s one Scan Context indexes (same split sc_relocalize.
        relocalize()'s docstring and _main() use: the SC descriptor and
        align_target's target cloud are different resolutions). gt_pose:
        always None here -- this bag layout carries no independent ground
        truth pose stream; kept as a seam for a future source that has one.
    """
    scans = sck.read_synced_scans(bag_path, cloud_topic, odom_topic)
    for i in range(0, len(scans), query_every):
        window = scans[:i + 1]
        if window[-1].stamp - window[0].stamp < tgt_accumulate_sec:
            continue  # not enough trailing history yet for a usable target
        ref_pose = scans[i].pose

        tgt_pts, _pose, _meta = sck.build_sc_cloud(
            window, ref_pose=ref_pose, accumulate_sec=tgt_accumulate_sec,
            voxel=0.0, ceiling_crop_m=None, max_translation_m=None)
        dense_target = o3d.geometry.PointCloud()
        dense_target.points = o3d.utility.Vector3dVector(tgt_pts)

        yield window, ref_pose, dense_target, None


# =============================================================================
# Seam 3 -- build the Scan Context query cloud
# =============================================================================
def build_sc_query_cloud(window, ref_pose, sc_params):
    """Thin wrapper over sc_keyframes.build_sc_cloud using the SAME recipe
    (accumulate_sec/voxel/ceiling_crop_m) the reference DB was built with
    (db.sc_params) -- query and database recipes must match exactly.
    Returns (pts, meta) -- meta's window_span_s is a direct diagnostic for
    "did this window actually get trimmed to accumulate_sec" that raw point
    extent can't give you on a long-range outdoor sensor (see run_verification)."""
    pts, _pose, meta = sck.build_sc_cloud(
        window, ref_pose=ref_pose,
        accumulate_sec=sc_params["accumulate_sec"],
        voxel=sc_params["voxel"],
        ceiling_crop_m=sc_params["ceiling_crop_m"])
    return pts, meta


# =============================================================================
# Seam 4 -- matcher query
# =============================================================================
def matcher_query(matcher, qpts):
    """(index, yaw_rad); index < 0 means no confident match. ScanContext
    Matcher.query() already returns exactly this shape -- normalize types
    only."""
    index, yaw_rad = matcher.query(qpts)
    return int(index), float(yaw_rad)


def matcher_query_raw(matcher, qpts):
    """DIAGNOSTIC companion to matcher_query -- (best_index, sc_dist,
    yaw_rad) for the nearest candidate regardless of sc_dist_thres (see
    ScanContextMatcher.query_raw). Lets a no_match row show the distance
    that WOULD have been returned, so "just missed the threshold" can be
    told apart from "genuinely nothing nearby"."""
    index, dist, yaw_rad = matcher.query_raw(qpts)
    return int(index), float(dist), float(yaw_rad)


# =============================================================================
# Seam 5 -- align_target params
# =============================================================================
def load_align_params(db):
    """The AlignParams instance align_target/build_sc_cloud expect.
    KeyframeDB already builds one (its own PreparedReference objects are
    prepared against it) -- reuse the same instance rather than
    constructing a second, possibly-divergent one."""
    return db.align_params


# =============================================================================
# Seam 6 -- align_target call + info-key extraction
# =============================================================================
def run_align(dense_target, prepared_ref, params, faiss_res):
    """(T_full, {coverage, rmse, teaser_yaw_deg}). Conservative path only
    (skip_teaser=False) -- see module docstring."""
    T_full, info = align_target(dense_target, prepared_ref, params,
                                 faiss_res=faiss_res, skip_teaser=False)
    return T_full, {
        "coverage": info["coverage"],
        "rmse": info["inlier_rmse"],
        "teaser_yaw_deg": info["teaser_yaw_deg"],
    }


# =============================================================================
# preflight -- confirm the three unknowns before the (possibly long) run
# =============================================================================
def preflight_checks(bag_path, cloud_topic, odom_topic, db, yaw_sign):
    bag = rosbag.Bag(bag_path, "r")
    try:
        topics = bag.get_type_and_topic_info().topics
    finally:
        bag.close()
    missing = [t for t in (cloud_topic, odom_topic) if t not in topics]
    if missing:
        raise SystemExit(
            "verify_pipeline: query bag is missing topic(s) %s -- got %s. "
            "Check --cloud_topic/--odom_topic against the actual bag."
            % (missing, sorted(topics.keys())))

    print("Preflight:")
    print("  query topics OK: %s (%d msgs), %s (%d msgs)"
          % (cloud_topic, topics[cloud_topic].message_count,
             odom_topic, topics[odom_topic].message_count))
    print("  align params: %s (defaults, not loaded from a config file -- "
          "see alignment_core.AlignParams)" % type(db.align_params).__name__)
    print("  yaw_sign=%+d -- confirm against "
          "scancontext/tests/test_yaw_convention.py's PASS output before "
          "trusting the yaw-agreement gate below" % yaw_sign)


# =============================================================================
# fixed orchestration
# =============================================================================
def run_verification(ref_dir, bag_path, yaw_sign, yaw_tol_deg, min_coverage,
                      max_rmse, min_dyaw_deg, query_every, cloud_topic,
                      odom_topic, tgt_accumulate_sec):
    matcher, db = load_reference(ref_dir)
    params = load_align_params(db)
    faiss_res = make_faiss_resources()

    preflight_checks(bag_path, cloud_topic, odom_topic, db, yaw_sign)

    records = []
    counts = Counter()
    for window, ref_pose, dense_target, _gt_pose in iterate_queries(
            bag_path, cloud_topic, odom_topic, query_every, tgt_accumulate_sec):
        q_stamp = window[-1].stamp

        qpts, sc_meta = build_sc_query_cloud(window, ref_pose, db.sc_params)
        index, yaw_rad = matcher_query(matcher, qpts)

        # ---- diagnostics computed for EVERY query, regardless of outcome --
        # these don't feed the gates, they explain why the gates fired.
        nearest_idx, nearest_m = nearest_kf_by_pose(db, ref_pose)
        best_idx, sc_dist, _best_yaw_rad = matcher_query_raw(matcher, qpts)
        q_n = len(qpts)
        # per-axis XY span (not np.ptp(qpts[:, :2]) with no axis -- that
        # flattens X and Y together into one meaningless number). On a
        # long-range outdoor sensor this can be large even for a single
        # viewpoint, so window_span_s (actual time covered after
        # accumulate_sec trimming) is the more reliable smear indicator.
        q_extent = float(np.ptp(qpts[:, :2], axis=0).max()) if q_n else 0.0
        q_window_s = sc_meta["window_span_s"]
        diag = dict(nearest_idx=nearest_idx, nearest_m=nearest_m,
                    best_idx=best_idx, sc_dist=sc_dist,
                    q_n=q_n, q_extent=q_extent, q_window_s=q_window_s)

        if index < 0:
            counts["no_match"] += 1
            records.append(dict(q=q_stamp, index=-1, sc_yaw=None,
                                 teaser_yaw=None, dyaw=None, coverage=None,
                                 rmse=None, result="no_match", **diag))
            continue

        # yaw seed -- built for completeness (this is what Seam B's fast
        # path would use) but NOT passed to align_target below: the
        # conservative path ignores init_T unless skip_teaser=True.
        T_init = make_T_yaw(yaw_sign * yaw_rad)
        sc_yaw_deg = yaw_from_T(T_init)

        prepared_ref, _ref_kf_pose = db[index]
        try:
            _T_full, info = run_align(dense_target, prepared_ref, params, faiss_res)
        except Exception as e:
            counts["align_error"] += 1
            records.append(dict(q=q_stamp, index=index, sc_yaw=sc_yaw_deg,
                                 teaser_yaw=None, dyaw=None, coverage=None,
                                 rmse=None, result="align_error: %s" % e, **diag))
            continue

        coverage, rmse, teaser_yaw = (info["coverage"], info["rmse"],
                                       info["teaser_yaw_deg"])
        dyaw = wrap_deg(teaser_yaw - sc_yaw_deg)

        if (not math.isfinite(rmse) or coverage < min_coverage
                or rmse > max_rmse or abs(dyaw) < min_dyaw_deg):
            result = "quality_gate"
        elif abs(dyaw) > yaw_tol_deg:
            result = "yaw_disagreement"
        else:
            result = "accepted"
        counts[result] += 1

        records.append(dict(q=q_stamp, index=index, sc_yaw=sc_yaw_deg,
                             teaser_yaw=teaser_yaw, dyaw=dyaw,
                             coverage=coverage, rmse=rmse, result=result, **diag))

    return records, counts


# =============================================================================
# --self_check mode -- fastest bisection, no bag needed
# =============================================================================
def self_check(db):
    """Query the matcher with each keyframe's OWN cloud (already in memory
    via db.prepared[k].ref_cloud -- no need to re-read .pcds from disk).
    Perfect retrieval => diagonal (k -> k, dist ~ 0). This isolates the
    retrieval core (tree/exclusion/keys) from everything downstream
    (recipe drift, GICP, gates) -- run this BEFORE debugging real queries."""
    ok = 0
    for k in range(len(db)):
        pts = np.asarray(db.prepared[k].ref_cloud.points)
        idx, dist, _yaw = db.matcher.query_raw(pts)
        hit = (idx == k)
        ok += int(hit)
        print("kf %3d -> idx %3d  dist %.4f  %s" % (k, idx, dist, "OK" if hit else "MISS"))
    print("\nself-check: %d/%d keyframes retrieve themselves" % (ok, len(db)))
    return ok, len(db)


# =============================================================================
# report
# =============================================================================
INTERPRETATION_NOTES = """\
- **yaw_disagreement**, clustered around one consistent sign flip -> wrong
  `--yaw_sign`; re-run `scancontext/tests/test_yaw_convention.py` and match
  its printed convention.
- **quality_gate** -> retrieval is likely landing on the right keyframe but
  GICP refinement is weak; tune `ceiling_crop_m` / `pc_max_radius` / keyframe
  spacing in the reference DB build (`sc_reference.py`).
- **no_match** -> the query is out-of-map, the reference DB is too sparse,
  or the query recipe (`accumulate_sec`/`voxel`/`ceiling_crop_m`) does not
  match the recipe the DB was built with (`db.sc_params`).
- **align_error** -> align_target raised (e.g. too few FPFH correspondences
  on a sparse/degenerate target window); same failure mode
  `sc_relocalize._main()`'s `--relocalize` guard already anticipates.

Diagnostic columns (`nearest_idx`/`nearest_m`/`best_idx`/`sc_dist`/`q_n`/`q_extent`)
explain WHY, independent of the gates above:
- `no_match` with small `nearest_m` (a keyframe really was right there) but
  large `sc_dist` -> the descriptors genuinely don't agree at the same
  place -> the query recipe likely differs from the DB build recipe
  (compare `db.sc_params` to what `iterate_queries`/`build_sc_query_cloud`
  apply -- accumulate_sec/voxel/ceiling_crop_m must match exactly).
- `no_match` with `sc_dist` just above the DB's `sc_dist_thres` -> the
  threshold is too strict for this scene; loosen `--sc_dist_thres` when
  rebuilding the DB (`sc_reference.py`).
- `no_match` with large `nearest_m` -> genuinely out-of-map; expected.
- `best_idx` != `nearest_idx` on a low-`sc_dist` row -> retrieval landed on
  a descriptor-similar but spatially-wrong keyframe (a symmetry/aliasing
  trap, not a recipe bug).
- `q_window_s` noticeably larger than `accumulate_sec` (config above) ->
  the window is NOT being trimmed correctly -- it's spanning real travel
  (a trajectory smear, not one viewpoint) -- a guaranteed no_match
  regardless of anything else. `q_extent` (raw XY span) is a weaker signal
  for this on a long-range outdoor sensor, where even a single viewpoint
  can show tens-to-hundreds of metres of extent from real sensor range;
  trust `q_window_s` over `q_extent` here. Run with `--self_check` first: a
  clean diagonal (every keyframe retrieves itself at ~0 distance) rules out
  the retrieval core (tree/exclusion/keys) entirely and confirms the
  problem is recipe/window drift between DB build and query, not the
  matcher itself.
"""


def _fmt(v, fmt="%.2f"):
    return "-" if v is None else (fmt % v)


def write_report(out_path, records, counts, config):
    total = len(records)
    accepted = counts.get("accepted", 0)
    accept_rate = (100.0 * accepted / total) if total else 0.0

    lines = []
    lines.append("# verify_pipeline findings\n")

    lines.append("## Summary\n")
    lines.append("- total queries: %d" % total)
    lines.append("- accepted: %d (%.0f%%)" % (accepted, accept_rate))
    for reason in ("no_match", "quality_gate", "yaw_disagreement", "align_error"):
        if counts.get(reason):
            lines.append("- %s: %d" % (reason, counts[reason]))
    lines.append("")

    lines.append("## Config\n")
    for k, v in config.items():
        lines.append("- %s: %s" % (k, v))
    lines.append("")

    lines.append("## Per-query results\n")
    lines.append("| q | index | sc_yaw | teaser_yaw | dyaw | coverage | rmse | result "
                 "| nearest_idx | nearest_m | best_idx | sc_dist | q_n | q_extent | q_window_s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append("| %.2f | %d | %s | %s | %s | %s | %s | %s "
                     "| %d | %s | %d | %s | %d | %s | %s |" % (
            r["q"], r["index"], _fmt(r["sc_yaw"]), _fmt(r["teaser_yaw"]),
            _fmt(r["dyaw"]), _fmt(r["coverage"]), _fmt(r["rmse"], "%.3f"),
            r["result"], r["nearest_idx"], _fmt(r["nearest_m"]),
            r["best_idx"], _fmt(r["sc_dist"], "%.4f"),
            r["q_n"], _fmt(r["q_extent"]), _fmt(r["q_window_s"])))
    lines.append("")

    lines.append("## Interpretation notes\n")
    lines.append(INTERPRETATION_NOTES)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("Wrote %s (%d queries, %d accepted)" % (out_path, total, accepted))


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Offline verification of the Scan Context -> "
                     "align_target pipeline against a revisit bag.")
    ap.add_argument("--ref_dir", required=True,
                     help="reference DB dir produced by sc_reference.py "
                          "(manifest.json + sc_params.json + keyframes/)")
    ap.add_argument("--bag", default=None,
                     help="revisit-pass query bag (not needed with --self_check)")
    ap.add_argument("--out", default="findings.md")
    ap.add_argument("--self_check", action="store_true",
                     help="skip the bag entirely: query the matcher with "
                          "each keyframe's OWN cloud and report whether it "
                          "retrieves itself. Fastest way to tell 'the "
                          "retrieval core is broken' apart from 'the query "
                          "recipe drifted from the DB build recipe' -- run "
                          "this first when real queries all miss.")

    ap.add_argument("--cloud_topic", default="/cloud_registered_body")
    ap.add_argument("--odom_topic", default="/Odometry")
    ap.add_argument("--query_every", type=int, default=5,
                     help="query every Nth synced scan")
    ap.add_argument("--tgt_accumulate_sec", type=float, default=1.5,
                     help="dense target window for align_target (see "
                          "sc_relocalize._main's TGT_ACCUMULATE_SEC)")

    ap.add_argument("--yaw_sign", type=float, choices=(1.0, -1.0), default=1.0,
                     help="sign to apply to the raw SC yaw_rad -- set from "
                          "tests/test_yaw_convention.py's PASS output")
    ap.add_argument("--yaw_tol_deg", type=float, default=30.0,
                     help="max |teaser_yaw - sc_yaw| to accept")
    ap.add_argument("--min_coverage", type=float, default=0.25)
    ap.add_argument("--max_rmse", type=float, default=0.3)
    ap.add_argument("--min_dyaw_deg", type=float, default=0.5,
                     help="reject if |teaser_yaw - sc_yaw| is under this "
                          "(quality_gate, not the yaw_disagreement gate)")
    args = ap.parse_args()

    if args.self_check:
        db = KeyframeDB(args.ref_dir)
        ok, n = self_check(db)
        raise SystemExit(0 if ok == n else 1)

    if not args.bag:
        ap.error("--bag is required (unless using --self_check)")

    records, counts = run_verification(
        args.ref_dir, args.bag, args.yaw_sign, args.yaw_tol_deg,
        args.min_coverage, args.max_rmse, args.min_dyaw_deg, args.query_every,
        args.cloud_topic, args.odom_topic, args.tgt_accumulate_sec)

    config = {
        "ref_dir": os.path.abspath(os.path.expanduser(args.ref_dir)),
        "bag": os.path.abspath(os.path.expanduser(args.bag)),
        "cloud_topic": args.cloud_topic,
        "odom_topic": args.odom_topic,
        "query_every": args.query_every,
        "tgt_accumulate_sec": args.tgt_accumulate_sec,
        "yaw_sign": args.yaw_sign,
        "yaw_tol_deg": args.yaw_tol_deg,
        "min_coverage": args.min_coverage,
        "max_rmse": args.max_rmse,
        "min_dyaw_deg": args.min_dyaw_deg,
    }
    write_report(args.out, records, counts, config)


if __name__ == "__main__":
    main()
