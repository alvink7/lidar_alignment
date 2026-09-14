#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alignment_node_multi.py
-----------------------
Drop-in alternative to alignment_node.py. Instead of aligning once at a fixed
num_frames, it aligns at SEVERAL accumulation depths (25/50/75/100 by default)
and publishes the single best candidate, chosen by median NN distance.

Same topics, same parameter names, same output as alignment_node.py -- existing
launch files, viz_test.py and any TF consumer need no change.

WHY BEST-OF-N AND NOT MORE FRAMES
    Measured on the indoor pair (reference_upscaled / test_target):

        N=25   yaw  +0.42 deg   cov 0.63   median 0.22    wrong
        N=50   yaw  -4.46 deg   cov 0.42   median 0.36    wrong
        N=75   yaw  -7.87 deg   cov 0.51   median 0.29    wrong
        N=100  yaw -88.14 deg   cov 0.90   median 0.08    CORRECT

    The three wrong answers are not near-misses converging on the right one --
    they are 80-89 deg away, parked in the rectangular room's 90 deg symmetry
    trap (session4 sec 4.7), creeping WITHIN that wrong basin. N=100 jumped
    basins. So:

      * there is no frame count that is right in general -- the same buffer
        depth that failed on this pair may be the one that works on another;
      * inter-checkpoint STABILITY is the wrong signal here. Runs 1-3 look far
        more mutually consistent than the jump to the correct answer. Stability
        detects "the solver stopped moving", and on a symmetric scene that is
        precisely what a wrong basin looks like. This node does NOT select on
        stability; it reports it, and flags when the winner disagrees with the
        crowd (which is what correctness looked like on this pair).

WHY median_dist AND NOT coverage OR rmse
    On those four runs:
      median_dist  0.08 vs 0.22 best-wrong -- a 2.75x gap, and UNCONDITIONAL
                   (every sampled point counts).
      coverage     0.90 vs 0.63 -- only 1.43x, and session4 sec 4.6 records
                   coverage 0.78 on a CONFIRMED-WRONG outdoor transform, so any
                   absolute threshold tuned on this pair sits within 0.02 of a
                   known false positive.
      inlier_rmse  0.11 vs 0.15 -- near noise, and structurally degenerate: it
                   averages only over points that already passed the inlier
                   test, so FEWER, LUCKIER inliers score BETTER. Never gate on
                   it.

    Selection is RELATIVE (best of four candidates on one pair), which is a
    weaker claim than an absolute threshold and is why this is defensible at all
    -- it needs no per-site calibration. It is still a heuristic over a metric
    session4 sec 4.6 documents as unreliable. The node prints the margin over
    the runner-up so you can see whether the choice was decisive or a coin flip,
    and it still honours ~min_coverage / ~max_rmse_m as publish gates.

    _select_metric:=med_ag scores with the GROUND REMOVED. Levelling puts both
    floors at Z=0, so floor points match under ANY planar transform -- that
    fraction is free and it inflates every ground-included score. If cov and
    cov_ag diverge widely in the table below, prefer med_ag.

USAGE
    rosrun cloud_aligner alignment_node_multi.py \
        _reference_pcd:=$HOME/catkin_ws/data/maps/droneparkscan.pcd \
        _input_topic:=/cloud_registered

    _frame_counts:=25,50,75,100   accumulation depths to try (nested)
    _select_metric:=median_dist   median_dist | med_ag | coverage | p90_ag
    _bag:=$HOME/bags/your.bag     read the topic from a bag instead of the wire
                                  (deterministic -- no connection race)
    _min_coverage:=0.0            publish gate on the WINNER
    _max_rmse_m:=1e9              publish gate on the WINNER
    _save_dir:=~/sweep            dump acc_NNNN.pcd per candidate
    _extra_metrics:=false         skip ground-removed scoring (faster)
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

import open3d as o3d

from alignment_core import (AlignParams, PreparedReference, align_target,
                            level_cloud, _remove_ground)
from ros_cloud_utils import (pointcloud2_to_o3d, accumulate_clouds,
                             matrix_to_transform_stamped)

try:
    import faiss
except ImportError:
    faiss = None

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# +1 = higher is better, -1 = lower is better
DIRECTION = {"median_dist": -1, "med_ag": -1, "p90_ag": -1, "coverage": +1}


def yaw_deg(T):
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def wrap_deg(a):
    return float((a + 180.0) % 360.0 - 180.0)


def subsample(pts, n, seed=0):
    if len(pts) <= n:
        return pts
    return pts[np.random.RandomState(seed).choice(len(pts), n, replace=False)]


class NN(object):
    def __init__(self, pcd):
        self.pts = np.asarray(pcd.points)
        self.tree = cKDTree(self.pts) if _HAVE_SCIPY else \
            o3d.geometry.KDTreeFlann(pcd)

    def dists(self, T, pts):
        q = (T[:3, :3] @ pts.T).T + T[:3, 3]
        if _HAVE_SCIPY:
            d, _ = self.tree.query(q, k=1)
            return np.asarray(d)
        out = np.full(len(q), np.inf)
        for i, x in enumerate(q):
            if np.all(np.isfinite(x)):
                k, _i, d2 = self.tree.search_knn_vector_3d(x, 1)
                if k > 0:
                    out[i] = np.sqrt(d2[0])
        return out


def prepare_levelled(pcd, params):
    icp = pcd.voxel_down_sample(params.ICP_VOXEL)
    icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=params.NORMAL_RADIUS, max_nn=30))
    return level_cloud(icp, params)


class MultiFrameAlignmentNode(object):

    def __init__(self):
        rospy.init_node("cloud_alignment_node_multi")

        # -------- I/O parameters (identical names to alignment_node.py) -----
        self.reference_pcd_path = os.path.expanduser(rospy.get_param(
            "~reference_pcd", "~/catkin_ws/data/maps/reference_upscaled.pcd"))
        self.input_topic = rospy.get_param("~input_topic", "/cloud_registered")
        self.output_topic = rospy.get_param("~output_topic",
                                            "/target_to_reference")
        self.matrix_topic = rospy.get_param("~matrix_topic",
                                            "/target_to_reference_matrix")
        self.reference_frame = rospy.get_param("~reference_frame",
                                               "reference_map")
        self.target_frame_override = rospy.get_param("~target_frame", "")
        self.broadcast_tf = bool(rospy.get_param("~broadcast_tf", True))
        self.odom_parent_frame = rospy.get_param("~odom_parent_frame",
                                                 "camera_init")

        # -------- multi-candidate parameters --------------------------------
        counts = str(rospy.get_param("~frame_counts", "25,50,75,100"))
        self.counts = sorted({int(c) for c in counts.replace(" ", "").split(",")
                              if c})
        self.select_metric = str(rospy.get_param("~select_metric",
                                                 "median_dist"))
        if self.select_metric not in DIRECTION:
            rospy.logfatal("~select_metric must be one of %s",
                           ", ".join(sorted(DIRECTION)))
            rospy.signal_shutdown("bad select_metric")
            return
        self.extra_metrics = bool(rospy.get_param("~extra_metrics", True))
        if self.select_metric in ("med_ag", "p90_ag"):
            self.extra_metrics = True          # selection depends on it
        self.sample_n = int(rospy.get_param("~sample", 3000))
        self.frame_timeout = float(rospy.get_param("~frame_timeout", 60.0))
        self.bag_path = os.path.expanduser(str(rospy.get_param("~bag", "")))
        self.save_dir = self._resolve_save_dir()

        # -------- publish gates (applied to the WINNER) ---------------------
        self.min_coverage = float(rospy.get_param("~min_coverage", 0.0))
        self.max_rmse_m = float(rospy.get_param("~max_rmse_m", 1e9))

        # -------- alignment parameters --------------------------------------
        self.params = AlignParams()
        for name in ("ICP_VOXEL", "NORMAL_RADIUS", "RANSAC_VOXEL",
                     "NORMAL_RADIUS_R", "FEATURE_RADIUS_R", "TEASER_NOISE_BOUND",
                     "TEASER_GNC_FACTOR", "ICP_DIST_COARSE", "ICP_DIST_FINE",
                     "GROUND_REMOVE_Z", "FLOOR_MAX_NORMAL_TILT_DEG",
                     "MIN_FLOOR_POINTS", "FLOOR_SEARCH_PCT", "FLOOR_SLAB_M",
                     "FLOOR_MIN_SLAB_FRAC", "FLOOR_BAND_M", "FLOOR_FIT_ITERS",
                     "FLOOR_FIT_SIGMA", "SCORE_INLIER_DIST"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, float(pv))
                rospy.loginfo("param override %s=%s", name, pv)
        for name in ("TEASER_MAX_ITER", "ICP_MAX_ITER"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, int(pv))
                rospy.loginfo("param override %s=%s", name, pv)

        # -------- load + prepare reference (once) ---------------------------
        if not os.path.isfile(self.reference_pcd_path):
            rospy.logfatal("Reference PCD not found: %s",
                           self.reference_pcd_path)
            rospy.signal_shutdown("no reference")
            return
        rospy.loginfo("Loading reference: %s", self.reference_pcd_path)
        ref_cloud = o3d.io.read_point_cloud(self.reference_pcd_path)
        if len(ref_cloud.points) == 0:
            rospy.logfatal("Reference PCD is empty.")
            rospy.signal_shutdown("empty reference")
            return
        rospy.loginfo("Reference points: %d. Caching (once)...",
                      len(ref_cloud.points))
        self.prepared_ref = PreparedReference(ref_cloud, self.params)
        self.ref_cloud = ref_cloud

        try:
            self.faiss_res = faiss.StandardGpuResources() if faiss else None
        except Exception:
            self.faiss_res = None

        # -------- state ------------------------------------------------------
        self.frames = []
        self.stamps = []
        self.lock = threading.Lock()
        self.rows = []
        self.target_frame_id = None
        self._last_target = None
        self.ref_lev = self.T_ref_level = None
        self.nn_full = self.nn_ag = None

        # -------- publishers -------------------------------------------------
        self.tf_pub = rospy.Publisher(self.output_topic, TransformStamped,
                                      queue_size=1, latch=True)
        self.mat_pub = rospy.Publisher(self.matrix_topic, Float64MultiArray,
                                       queue_size=1, latch=True)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # -------- input ------------------------------------------------------
        # Subscribe immediately (before any further prep). A later connection
        # means a different frame 0, which means a different accumulated cloud,
        # which means a different transform on identical code.
        self.sub = None
        if not self.bag_path:
            self.sub = rospy.Subscriber(
                self.input_topic, PointCloud2, self.cloud_cb,
                queue_size=int(rospy.get_param("~queue_size", 200)))
            rospy.loginfo("Listening on %s -- candidates at %s frames.",
                          self.input_topic,
                          ", ".join(str(c) for c in self.counts))
        else:
            rospy.loginfo("Bag mode: reading %s from %s",
                          self.input_topic, self.bag_path)

        threading.Thread(target=self.run_all, daemon=True).start()

    # -------------------------------------------------------------- saving --
    def _resolve_save_dir(self):
        """Resolve ~save_dir loudly. Silently doing nothing when the parameter
        did not arrive is how a confirmed-correct accumulation got lost: the
        transform was published, the cloud that produced it was not written, and
        the result became unreproducible."""
        # Accept the private param first, then a global /save_dir. A <param>
        # written just OUTSIDE the <node> element lands in the global namespace
        # instead of the node's private one -- easy to do when the node tag is
        # self-closing -- and the intent is unambiguous either way, so honour it
        # rather than silently writing nothing.
        raw = rospy.get_param("~save_dir", "")
        if not raw:
            raw = rospy.get_param("/save_dir", "")
            if raw:
                rospy.logwarn(
                    "using GLOBAL /save_dir=%r -- ~save_dir was not set. The "
                    "<param> is outside the <node> element; move it inside so "
                    "it is private to this node.", raw)
        if not raw:
            rospy.logwarn("~save_dir is NOT set -- accumulated clouds will NOT "
                          "be written, and any transform published here will "
                          "be unreproducible. Pass _save_dir:=$HOME/sweep")
            rospy.logwarn("  (with roslaunch, <param name=\"save_dir\" .../> "
                          "must be INSIDE the <node>...</node> element; a "
                          "self-closing <node ... /> puts any following "
                          "<param> in the global namespace. _name:=value is "
                          "rosrun-only syntax and roslaunch ignores it.)")
            rospy.logwarn("  this node resolved ~ to: %s",
                          rospy.get_name())
            try:
                near = [k for k in rospy.get_param_names()
                        if "save" in k.lower() or "dir" in k.lower()]
                if near:
                    rospy.logwarn("  params on the master that look related: "
                                  "%s", ", ".join(sorted(near)))
            except Exception:
                pass
            return ""
        path = os.path.abspath(os.path.expanduser(str(raw)))
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
            probe = os.path.join(path, ".write_test")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
        except Exception as e:
            rospy.logerr("~save_dir=%r is not writable (%s: %s) -- clouds will "
                         "NOT be saved.", path, type(e).__name__, e)
            return ""
        rospy.loginfo("save_dir: %s (writable)", path)
        return path

    def _save_cloud(self, cloud, n):
        """write_point_cloud returns False on failure and raises nothing. Check
        it, and confirm the file actually landed."""
        if not self.save_dir:
            return
        path = os.path.join(self.save_dir, "acc_%04d.pcd" % n)
        try:
            ok = o3d.io.write_point_cloud(path, cloud)
        except Exception as e:
            rospy.logerr("failed writing %s: %s: %s", path,
                         type(e).__name__, e)
            return
        if not ok or not os.path.isfile(path):
            rospy.logerr("write_point_cloud reported failure for %s", path)
            return
        rospy.loginfo("saved %s (%d pts, %.1f MB)", path, len(cloud.points),
                      os.path.getsize(path) / 1e6)

    # ------------------------------------------------------------- input ----
    def cloud_cb(self, msg):
        with self.lock:
            if len(self.frames) >= max(self.counts):
                return
        if self.target_frame_id is None:
            self.target_frame_id = (self.target_frame_override
                                    or msg.header.frame_id or "target")
        cloud = pointcloud2_to_o3d(msg)
        if len(cloud.points) == 0:
            rospy.logwarn_throttle(2.0, "empty cloud frame skipped")
            return
        with self.lock:
            self.frames.append(cloud)
            self.stamps.append(msg.header.stamp)
            n = len(self.frames)
        rospy.loginfo_throttle(1.0, "accumulated %d/%d frames",
                               n, max(self.counts))

    def _load_bag(self):
        import rosbag
        with rosbag.Bag(self.bag_path, "r") as bag:
            _types, tinfo = bag.get_type_and_topic_info()
            if self.input_topic not in tinfo:
                rospy.logfatal("%s not in %s. Present: %s", self.input_topic,
                               self.bag_path, ", ".join(sorted(tinfo.keys())))
                return False
            rospy.loginfo("%s: %d messages; reading the first %d",
                          self.input_topic,
                          tinfo[self.input_topic].message_count,
                          max(self.counts))
            for _t, msg, _ts in bag.read_messages(topics=[self.input_topic]):
                if self.target_frame_id is None:
                    self.target_frame_id = (self.target_frame_override
                                            or msg.header.frame_id or "target")
                cloud = pointcloud2_to_o3d(msg)
                if len(cloud.points) == 0:
                    continue
                self.frames.append(cloud)
                self.stamps.append(msg.header.stamp)
                if len(self.frames) >= max(self.counts):
                    break
        rospy.loginfo("read %d frames from bag", len(self.frames))
        return len(self.frames) > 0

    def _wait_for(self, n):
        if self.bag_path:
            return len(self.frames) >= n
        t0 = time.time()
        while not rospy.is_shutdown():
            with self.lock:
                have = len(self.frames)
            if have >= n:
                return True
            if time.time() - t0 > self.frame_timeout:
                rospy.logwarn("timed out waiting for %d frames (have %d) -- "
                              "input ended; using the candidates so far", n, have)
                return False
            time.sleep(0.05)
        return False

    def _prep_extra(self):
        if not self.extra_metrics:
            return
        t0 = time.time()
        self.ref_lev, self.T_ref_level, meta = prepare_levelled(
            self.ref_cloud, self.params)
        ref_ag = _remove_ground(self.ref_lev, self.params.GROUND_REMOVE_Z)
        self.nn_full, self.nn_ag = NN(self.ref_lev), NN(ref_ag)
        rospy.loginfo("reference levelled + KD-trees in %.1fs (floor %d pts, "
                      "tilt %.2fdeg, ground_z %.3f)", time.time() - t0,
                      meta["n_floor"], meta["tilt_deg"], meta["ground_z"])

    # --------------------------------------------------------- candidates ----
    def run_all(self):
        if self.bag_path and not self._load_bag():
            rospy.signal_shutdown("no frames")
            return
        self._prep_extra()

        for n in self.counts:
            if not self._wait_for(n):
                break
            with self.lock:
                batch = list(self.frames[:n])
                stamp = self.stamps[n - 1]
            self.run_one(n, batch, stamp)

        self.select_and_publish()
        if self.sub is not None:
            self.sub.unregister()

    def run_one(self, n, batch, stamp):
        rospy.loginfo("--- candidate N=%d ---", n)
        target = accumulate_clouds(batch)
        self._last_target = target
        rospy.loginfo("Merged target points: %d", len(target.points))
        self._save_cloud(target, n)

        t0 = time.time()
        try:
            T, info = align_target(
                target, self.prepared_ref, self.params,
                faiss_res=self.faiss_res,
                logger=lambda m: rospy.loginfo("  [align] %s", m))
        except Exception as e:
            rospy.logerr("N=%d alignment failed: %s: %s", n, type(e).__name__, e)
            return
        secs = time.time() - t0

        row = dict(frames=n, T=T, stamp=stamp, secs=secs,
                   yaw=yaw_deg(T), tx=float(T[0, 3]), ty=float(T[1, 3]),
                   **{k: info.get(k) for k in (
                       "coverage", "inlier_rmse", "median_dist",
                       "n_correspondences", "teaser_inliers")})
        row.update(self.score_extra(T))
        self.rows.append(row)
        rospy.loginfo("N=%d: yaw=%.2fdeg t=[%.3f, %.3f] cov=%.3f rmse=%.3f "
                      "median=%.3f%s (%.1fs)", n, row["yaw"], row["tx"],
                      row["ty"], row["coverage"], row["inlier_rmse"],
                      row["median_dist"],
                      "" if "med_ag" not in row
                      else " med_ag=%.3f" % row["med_ag"], secs)

    def score_extra(self, T_full):
        """Ground-removed rescoring. Both floors sit at Z=0 after levelling, so
        every ground point matches under any planar transform; whatever fraction
        of the target is ground is coverage the transform did not earn."""
        if not self.extra_metrics or self._last_target is None:
            return {}
        try:
            tgt_lev, T_tgt_level, _meta = prepare_levelled(self._last_target,
                                                           self.params)
            T_lev = self.T_ref_level @ T_full @ np.linalg.inv(T_tgt_level)
            tgt_ag = _remove_ground(tgt_lev, self.params.GROUND_REMOVE_Z)
            out = {"ground_frac": 1.0 - len(tgt_ag.points)
                   / max(len(tgt_lev.points), 1)}
            if len(tgt_ag.points):
                samp = subsample(np.asarray(tgt_ag.points), self.sample_n)
                d = self.nn_ag.dists(T_lev, samp)
                out["cov_ag"] = float((d < self.params.SCORE_INLIER_DIST).mean())
                out["med_ag"] = float(np.median(d))
                out["p90_ag"] = float(np.percentile(d, 90))
            return out
        except Exception as e:
            rospy.logwarn("supplementary scoring failed: %s", e)
            return {}

    # ------------------------------------------------------------ select ----
    def select_and_publish(self):
        m = self.select_metric
        usable = [r for r in self.rows
                  if r.get(m) is not None and np.isfinite(r.get(m, np.nan))]
        if not usable:
            rospy.logerr("No candidate produced a usable %s. Not publishing.", m)
            return

        rospy.loginfo("=== CANDIDATES ===")
        rospy.loginfo("%6s %9s %9s %9s | %7s %7s %8s | %7s %7s | %6s",
                      "frames", "yaw", "tx", "ty", "cov", "rmse", "median",
                      "cov_ag", "med_ag", "secs")
        for r in self.rows:
            rospy.loginfo("%6d %9.2f %9.3f %9.3f | %7.3f %7.3f %8.3f | "
                          "%7s %7s | %6.1f",
                          r["frames"], r["yaw"], r["tx"], r["ty"],
                          r["coverage"], r["inlier_rmse"], r["median_dist"],
                          "%.3f" % r["cov_ag"] if "cov_ag" in r else "-",
                          "%.3f" % r["med_ag"] if "med_ag" in r else "-",
                          r["secs"])

        ordered = sorted(usable, key=lambda r: -DIRECTION[m] * r[m])
        best = ordered[0]
        runner = ordered[1] if len(ordered) > 1 else None

        rospy.loginfo("=== SELECTED: N=%d frames (best %s = %.4f) ===",
                      best["frames"], m, best[m])
        if runner is not None:
            if DIRECTION[m] < 0:
                factor = runner[m] / max(best[m], 1e-9)
            else:
                factor = best[m] / max(runner[m], 1e-9)
            rospy.loginfo("margin over runner-up (N=%d, %s=%.4f): %.2fx",
                          runner["frames"], m, runner[m], factor)
            if factor < 1.25:
                rospy.logwarn("MARGIN IS THIN (%.2fx). The winner is barely "
                              "distinguishable from the runner-up on a metric "
                              "session4 sec 4.6 documents as unreliable. Treat "
                              "this result as unconfirmed and check the "
                              "overlay.", factor)

        # Does the winner agree with the others, or is it the odd one out?
        # On the indoor pair the CORRECT answer was the odd one out (three wrong
        # runs agreed with each other inside the 90 deg symmetry trap), so this
        # is reported, never used to override the metric.
        others = [r for r in self.rows if r is not best]
        if others:
            agree = [r for r in others
                     if abs(wrap_deg(r["yaw"] - best["yaw"])) < 5.0
                     and np.hypot(r["tx"] - best["tx"],
                                  r["ty"] - best["ty"]) < 1.0]
            if not agree:
                rospy.logwarn("the winner disagrees with EVERY other candidate "
                              "(yaw spread %.1f deg). That is not itself a "
                              "problem -- on the indoor pair the correct answer "
                              "was exactly this shape -- but it does mean "
                              "nothing corroborates it. Confirm with an "
                              "overlay before trusting it.",
                              max(abs(wrap_deg(r["yaw"] - best["yaw"]))
                                  for r in others))
            else:
                rospy.loginfo("%d of %d other candidates agree with the winner "
                              "within 5 deg / 1 m", len(agree), len(others))

        if "cov_ag" in best and best["coverage"] > 0.5 and best["cov_ag"] < 0.25:
            rospy.logwarn("winner coverage %.2f collapses to %.2f with the "
                          "ground removed (%.0f%% of the target is floor). The "
                          "headline score is floor, not structure -- consider "
                          "_select_metric:=med_ag.", best["coverage"],
                          best["cov_ag"], 100.0 * best.get("ground_frac", 0.0))

        T_full = best["T"]
        np.set_printoptions(precision=4, suppress=True)
        rospy.loginfo("T (target -> reference):\n%s", str(T_full))

        # Write every candidate's matrix beside its cloud. A transform without
        # the cloud that produced it cannot be reproduced or re-scored later.
        if self.save_dir:
            for r in self.rows:
                p = os.path.join(self.save_dir, "T_%04d.txt" % r["frames"])
                np.savetxt(p, r["T"], fmt="%.9f")
            sel = os.path.join(self.save_dir, "T_selected.txt")
            np.savetxt(sel, T_full, fmt="%.9f")
            rospy.loginfo("wrote %s (from N=%d) and per-candidate T_NNNN.txt",
                          sel, best["frames"])

        # ---- publish gates, applied to the winner only ----
        cov, rmse = best["coverage"], best["inlier_rmse"]
        if (not np.isfinite(rmse)) or (cov < self.min_coverage) or \
           (rmse > self.max_rmse_m):
            rospy.logerr("REFUSING to publish: coverage=%.2f (min %.2f) "
                         "rmse=%.2fm (max %.2fm).", cov, self.min_coverage,
                         rmse, self.max_rmse_m)
            rospy.loginfo("Done (no publish). Node will idle; Ctrl-C to exit.")
            return

        self.publish(T_full, best["stamp"])

    def publish(self, T_full, stamp):
        target_frame = self.target_frame_id or "target"

        ts = matrix_to_transform_stamped(T_full, self.reference_frame,
                                         target_frame, stamp)
        self.tf_pub.publish(ts)

        mat = Float64MultiArray()
        mat.layout.dim = [
            MultiArrayDimension(label="rows", size=4, stride=16),
            MultiArrayDimension(label="cols", size=4, stride=4),
        ]
        mat.data = [float(x) for x in T_full.flatten()]
        self.mat_pub.publish(mat)

        if self.broadcast_tf:
            static_ts = matrix_to_transform_stamped(
                T_full, self.reference_frame, self.odom_parent_frame, stamp)
            self.static_tf_broadcaster.sendTransform(static_ts)
            rospy.loginfo("Broadcast static TF: %s -> %s",
                          self.reference_frame, self.odom_parent_frame)

        rospy.loginfo("Published transform on %s and matrix on %s (latched).",
                      self.output_topic, self.matrix_topic)
        rospy.loginfo("Done. Node will idle; Ctrl-C to exit.")


if __name__ == "__main__":
    try:
        MultiFrameAlignmentNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass