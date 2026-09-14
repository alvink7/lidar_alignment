#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_align_stages.py -- Layer 1 stage timing for align_target.

No Scan Context, no bag reading, no retrieval. You give it two .pcd files
already accumulated from /cloud_registered (a target and a reference), and it
runs align_target many times to produce a stable per-stage timing table:
which stage (FPFH / correspondences / TEASER / GICP / level) eats the
wall-clock, its share of the total, and the Amdahl ceiling (max speedup if you
perfected that stage). Read this before writing any CUDA/threading.

Because the same two clouds align every iteration, "many iterations" measures
the compute distribution (warmup, jitter, GC), which is exactly what you want
for a profiling decision -- not input variation.

Correctness features:
  - GPU sync before stopping the correspondences timer (async kernels otherwise
    make the GPU look fake-fast and inflate the next stage).
  - Warmup discard (first calls include CUDA/FAISS/Open3D init).
  - Median (not mean) + min/max so data-dependent variance is visible.
  - Amdahl ceiling per stage.

Run:
    python profile_align_stages.py --target target.pcd --reference ref.pcd --n 60
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from alignment_core import AlignParams, PreparedReference
import alignment_core as AC


# ---------------------------------------------------------------------------
# GPU sync -- called before stopping any timer that launched GPU work.
# ---------------------------------------------------------------------------
_SYNC_BACKEND = (None, None)


def _detect_sync():
    global _SYNC_BACKEND
    try:
        import torch
        if torch.cuda.is_available():
            _SYNC_BACKEND = ("torch", torch)
            return
    except Exception:
        pass
    try:
        import cupy
        _SYNC_BACKEND = ("cupy", cupy)
        return
    except Exception:
        pass
    _SYNC_BACKEND = (None, None)


def _gpu_sync():
    kind, mod = _SYNC_BACKEND
    if kind == "torch":
        mod.cuda.synchronize()
    elif kind == "cupy":
        mod.cuda.runtime.deviceSynchronize()
    # else: no backend -> correspondences may be slightly under-counted


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class StageTimer:
    def __init__(self):
        self.samples = defaultdict(list)

    def add(self, label, ms):
        self.samples[label].append(ms)

    def report(self, warmup):
        rows = []
        for k, s in self.samples.items():
            s = s[warmup:] if len(s) > warmup else s
            if not s:
                continue
            a = np.asarray(s)
            rows.append(dict(stage=k, median=float(np.median(a)), mean=float(a.mean()),
                             mn=float(a.min()), mx=float(a.max()), n=len(a)))
        if not rows:
            print("no samples collected -- did align_target run?")
            return

        # fpfh and gicp fire more than once per align_target call (target+ref
        # fpfh; coarse+fine gicp loop). Infer calls-per-run from a stage that
        # fires exactly once (teaser); scale each stage's per-call contribution
        # by its firings-per-call so the %tot column reflects a single call.
        teaser_rows = [r for r in rows if r["stage"] == "teaser"]
        n_calls = teaser_rows[0]["n"] if teaser_rows else max(r["n"] for r in rows)

        for r in rows:
            firings_per_call = r["n"] / float(n_calls) if n_calls else 1.0
            r["per_call"] = r["median"] * firings_per_call
        total = sum(r["per_call"] for r in rows)

        rows.sort(key=lambda r: -r["per_call"])
        print("\n=== Layer 1: align_target stage timing "
              "(median per firing; %d calls) ===" % n_calls)
        print("%-18s %10s %9s %8s %8s %9s %7s %6s" %
              ("stage", "median_ms", "mean_ms", "min", "max", "per_call", "%tot", "n"))
        print("-" * 80)
        for r in rows:
            print("%-18s %10.2f %9.2f %8.2f %8.2f %9.2f %7.1f %6d" %
                  (r["stage"], r["median"], r["mean"], r["mn"], r["mx"],
                   r["per_call"], 100.0 * r["per_call"] / total, r["n"]))
        print("-" * 80)
        print("%-18s %38s %9.2f  (sum of stages, per call)" % ("TOTAL", "", total))

        print("\n--- Amdahl ceiling (max whole-call speedup if a stage -> 0) ---")
        for r in rows:
            frac = r["per_call"] / total
            speedup = 1.0 / (1.0 - frac) if frac < 1 else float("inf")
            print("  eliminate %-16s -> up to %5.2fx  (%.1f%% of call)"
                  % (r["stage"], speedup, 100 * frac))

        print("\n--- variance flags (max/median > 3 => data-dependent) ---")
        flagged = [r for r in rows if r["median"] > 0 and r["mx"] / r["median"] > 3]
        if flagged:
            for r in flagged:
                print("  %-16s max/median = %.1f" % (r["stage"], r["mx"] / r["median"]))
        else:
            print("  none -- stage costs are stable")


# ---------------------------------------------------------------------------
# monkeypatch the internal stage helpers so we time them without editing
# alignment_core. fpfh/gicp fire multiple times per call -- see report() note.
# ---------------------------------------------------------------------------
def install_timing(T):
    def wrap(name, fn, sync=False):
        def inner(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            if sync:
                _gpu_sync()
            T.add(name, (time.perf_counter() - t0) * 1e3)
            return r
        return inner

    AC._fpfh = wrap("fpfh", AC._fpfh)
    AC._correspondences = wrap("correspondences", AC._correspondences, sync=True)
    AC._teaser = wrap("teaser", AC._teaser)          # fires once/call
    AC._gicp = wrap("gicp", AC._gicp)
    if hasattr(AC, "level_cloud"):
        AC.level_cloud = wrap("level_cloud", AC.level_cloud)
    if hasattr(AC, "constrain_planar"):
        AC.constrain_planar = wrap("constrain_planar", AC.constrain_planar)
    # previously-unwrapped work inside align_target -- accounted for the gap
    # between whole-call time and the sum of the stages above.
    if hasattr(AC, "_prep_icp_cloud"):
        AC._prep_icp_cloud = wrap("prep_icp_cloud", AC._prep_icp_cloud)   # fires 2x/call (ref+tgt)
    if hasattr(AC, "_remove_ground"):
        AC._remove_ground = wrap("remove_ground", AC._remove_ground)      # fires 2x/call (ref+tgt)
    if hasattr(AC, "normalize_xy"):
        AC.normalize_xy = wrap("normalize_xy", AC.normalize_xy)           # fires 2x/call (ref+tgt)
    if hasattr(AC, "_score_alignment"):
        AC._score_alignment = wrap("score_alignment", AC._score_alignment)  # fires 1x/call


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(target_pcd, reference_pcd, n, warmup):
    _detect_sync()
    kind = _SYNC_BACKEND[0]
    print("GPU sync backend: %s" % (kind or "NONE (correspondences may be under-counted)"))

    tgt = o3d.io.read_point_cloud(os.path.expanduser(target_pcd))
    ref = o3d.io.read_point_cloud(os.path.expanduser(reference_pcd))
    if len(tgt.points) == 0 or len(ref.points) == 0:
        raise SystemExit("empty target or reference cloud")
    print("target: %d pts   reference: %d pts" % (len(tgt.points), len(ref.points)))

    params = AlignParams()
    prepared_ref = PreparedReference(ref, params)

    try:
        import faiss
        faiss_res = faiss.StandardGpuResources()
        print("faiss GPU resources: ON")
    except Exception as e:
        faiss_res = None
        print("faiss GPU resources: OFF (%s)" % e)

    T = StageTimer()
    install_timing(T)

    total = n + warmup
    print("\nrunning %d align_target calls (%d warmup + %d measured)...\n"
          % (total, warmup, n))
    t_wall0 = time.perf_counter()
    for i in range(total):
        t0 = time.perf_counter()
        AC.align_target(tgt, prepared_ref, params, faiss_res=faiss_res)
        T.add("__whole_call__", (time.perf_counter() - t0) * 1e3)
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, total))
    wall = time.perf_counter() - t_wall0
    print("\nwall time for %d calls: %.2f s (%.1f ms/call avg incl. warmup)"
          % (total, wall, 1e3 * wall / total))

    whole = np.asarray(T.samples.pop("__whole_call__")[warmup:])
    print("\nwhole align_target call: median %.2f ms  mean %.2f ms  min %.2f  max %.2f"
          % (np.median(whole), whole.mean(), whole.min(), whole.max()))

    T.report(warmup=warmup)

    print("\nNote: 'fpfh' and 'gicp' fire multiple times per call (target+ref FPFH; "
          "coarse+fine GICP loop). The %tot / per_call columns already scale by "
          "firings-per-call. To split target vs ref FPFH or coarse vs fine GICP, "
          "instrument align_target directly.")


def main():
    ap = argparse.ArgumentParser(
        description="Layer 1 stage timing for align_target (two PCDs, many iterations).")
    ap.add_argument("--target", required=True,
                    help="target .pcd (accumulated from /cloud_registered)")
    ap.add_argument("--reference", required=True, help="reference .pcd")
    ap.add_argument("--n", type=int, default=60, help="measured iterations (post-warmup)")
    ap.add_argument("--warmup", type=int, default=5, help="warmup iterations to discard")
    args = ap.parse_args()
    run(args.target, args.reference, args.n, args.warmup)


if __name__ == "__main__":
    main()