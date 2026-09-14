#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bisect
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

import open3d as o3d
import faiss

from alignment_core import AlignParams, PreparedReference, align_target
from ros_cloud_utils import (pointcloud2_to_o3d, accumulate_clouds,
                             matrix_to_transform_stamped)

# Scan Context (scancontext/python) is imported lazily in __init__, only when
# ~use_scan_context is set -- so the node has ZERO new hard deps in the
# default (baseline) configuration.

class AlignmentNode(object):
    def __init__(self):
        rospy.init_node("cloud_alignment_node")

        # -------- I/O parameters --------
        # No silent default: it lives in config/params.yaml. A baked-in fallback
        # here would quietly align against the WRONG map if the yaml failed to
        # load (the old default, reference_upscaled.pcd, exists on disk, so the
        # failure would look like plausible-but-wrong output rather than an
        # error). Empty -> the baseline branch below logfatals.
        self.reference_pcd_path = os.path.expanduser(
            rospy.get_param("~reference_pcd", ""))
        self.input_topic = rospy.get_param("~input_topic", "/cloud_registered")
        self.output_topic = rospy.get_param("~output_topic", "/target_to_reference")
        self.matrix_topic = rospy.get_param("~matrix_topic", "/target_to_reference_matrix")
        self.num_frames = int(rospy.get_param("~num_frames", 10))
        self.reference_frame = rospy.get_param("~reference_frame", "reference_map")
        # target frame: read from incoming msg header, or override
        self.target_frame_override = rospy.get_param("~target_frame", "")
        # static TF : broadcast reference_frame -> odom_parent_frame
        # so FAST-LIO's odom_parent -> body chain composes into the reference frame.
        self.broadcast_tf = bool(rospy.get_param("~broadcast_tf", True))
        # FAST-LIO's odometry parent frame (what its camera_init/odom is called).
        self.odom_parent_frame = rospy.get_param("~odom_parent_frame", "camera_init")

        # -------- Scan Context (optional reference selection) --------
        # When false (default) the node uses the fixed ~reference_pcd, exactly
        # as before -- everything below this block is a no-op in that case.
        self.use_scan_context = bool(rospy.get_param("~use_scan_context", False))
        self.sc_db_dir = os.path.expanduser(rospy.get_param("~sc_db_dir", ""))
        # egocentric (body-frame) cloud + pose, distinct from ~input_topic
        # (world-frame) -- Scan Context needs the same recipe the reference
        # DB was built from (see scancontext/python/sc_keyframes.py).
        self.sc_cloud_topic = rospy.get_param("~sc_cloud_topic", "/cloud_registered_body")
        self.sc_odom_topic = rospy.get_param("~sc_odom_topic", "/Odometry")
        self.sc_max_odom_gap = float(rospy.get_param("~sc_max_odom_gap", 0.05))
        # skip_teaser=False (default): full TEASER+GICP path -- SC only picks
        # the reference keyframe; yaw is independently re-derived by TEASER,
        # same as verify_pipeline.py's conservative cross-check. Set true to
        # skip TEASER/FPFH and seed GICP directly from the SC yaw instead
        # (must be explicitly turned on).
        self.sc_skip_teaser = bool(rospy.get_param("~sc_skip_teaser", False))

        # -------- Publish gate thresholds (optional; both default OFF) --------
        #   ~min_coverage : refuse to publish below this coverage
        #   ~max_rmse_m   : refuse to publish above this inlier RMSE (metres)
        self.min_coverage = float(rospy.get_param("~min_coverage", 0.0))
        self.max_rmse_m = float(rospy.get_param("~max_rmse_m", 1e9))

        # -------- Alignment parameters (global map-to-map registration) --------
        # align_target now runs the full 6-DOF geometry-only path (see
        # global_register.py): voxel downsample -> FPFH -> mutual-NN
        # correspondences -> TEASER++ (full 6-DOF) -> GICP refine. One knob,
        # ~GLOBAL_VOXEL, sets the sampling scale; everything else below is a
        # multiple of it. No floor-levelling, no planar constraint.
        #   ~GLOBAL_VOXEL      : sampling scale for the whole-map solve
        #   ~GLOBAL_NORMAL_R   : normal-estimation radius
        #   ~GLOBAL_FEATURE_R  : FPFH feature radius
        #   ~GLOBAL_TEASER_NB  : TEASER inlier/noise bound
        self.params = AlignParams()

        # GLOBAL_VOXEL is the ONE knob: AlignParams defines the other three
        # GLOBAL_* radii as multiples of it, but that derivation runs ONCE at
        # class-definition time. So overriding only ~GLOBAL_VOXEL here would
        # leave the other three at the values derived from the in-code default,
        # producing an INCOHERENT set -- e.g. voxel=0.75 with a stale 0.30
        # normal radius means every normal/FPFH descriptor is built from ~1
        # neighbour (pure noise) and the TEASER bound is far tighter than the
        # sampling scatter. That yields an arbitrary ~90deg-tilt transform with
        # coverage 0 -- a real failure that reached the field. Re-derive the
        # dependents whenever the voxel is overridden; an EXPLICIT per-radius
        # override still wins, since it is applied afterwards.
        voxel_override = rospy.get_param("~GLOBAL_VOXEL", None)
        if voxel_override is not None:
            v = float(voxel_override)
            self.params.GLOBAL_VOXEL = v
            self.params.GLOBAL_NORMAL_R = v * 3.0
            self.params.GLOBAL_FEATURE_R = v * 5.0
            self.params.GLOBAL_TEASER_NB = v * 2.0
            rospy.loginfo("param override GLOBAL_VOXEL=%.3f -> derived "
                          "NORMAL_R=%.3f FEATURE_R=%.3f TEASER_NB=%.3f",
                          v, self.params.GLOBAL_NORMAL_R,
                          self.params.GLOBAL_FEATURE_R,
                          self.params.GLOBAL_TEASER_NB)

        for name in ("GLOBAL_NORMAL_R", "GLOBAL_FEATURE_R",
                     "GLOBAL_TEASER_NB", "TEASER_GNC_FACTOR"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, float(pv))
                rospy.loginfo("param override %s=%s (explicit; wins over derived)",
                              name, pv)
        for name in ("TEASER_MAX_ITER", "ICP_MAX_ITER"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, int(pv))
                rospy.loginfo("param override %s=%s", name, pv)

        # Always log the FINAL scale set, and refuse to fail silently on an
        # incoherent one. A normal/feature radius at or below the voxel pitch
        # means each descriptor sees ~1 neighbour -> meaningless FPFH -> an
        # arbitrary transform with ~0 coverage (see the comment above).
        rospy.loginfo("global-reg scale: voxel=%.3f normal_r=%.3f feat_r=%.3f "
                      "teaser_nb=%.3f", self.params.GLOBAL_VOXEL,
                      self.params.GLOBAL_NORMAL_R, self.params.GLOBAL_FEATURE_R,
                      self.params.GLOBAL_TEASER_NB)
        if (self.params.GLOBAL_NORMAL_R <= self.params.GLOBAL_VOXEL or
                self.params.GLOBAL_FEATURE_R <= self.params.GLOBAL_VOXEL):
            rospy.logerr("INCOHERENT global-reg scale: normal_r=%.3f / "
                         "feat_r=%.3f are not larger than voxel=%.3f. FPFH will "
                         "be computed from ~1 neighbour and registration WILL "
                         "fail (arbitrary rotation, coverage ~0). Leave the "
                         "GLOBAL_NORMAL_R/GLOBAL_FEATURE_R/GLOBAL_TEASER_NB "
                         "params unset so they derive from ~GLOBAL_VOXEL.",
                         self.params.GLOBAL_NORMAL_R,
                         self.params.GLOBAL_FEATURE_R, self.params.GLOBAL_VOXEL)

        # -------- load reference (once) --------
        self.kf_db = None
        self.prepared_ref = None
        if self.use_scan_context:
            if not self.sc_db_dir or not os.path.isdir(self.sc_db_dir):
                rospy.logfatal("use_scan_context=true but ~sc_db_dir invalid: %s",
                               self.sc_db_dir)
                rospy.signal_shutdown("no sc db")
                return
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "scancontext", "python"))
            import sc_keyframes as sck
            from sc_matcher import ScanContextMatcher
            from sc_relocalize import KeyframeDB, relocalize
            self._sck = sck
            self._relocalize = relocalize

            rospy.loginfo("Loading Scan Context keyframe DB: %s", self.sc_db_dir)
            # align_params=self.params: the DB's PreparedReference objects and
            # any align_target call through it use the SAME (possibly
            # ROS-param-overridden) AlignParams as the baseline path.
            self.kf_db = KeyframeDB(self.sc_db_dir, align_params=self.params)
            rospy.loginfo("SC DB loaded: %d keyframes (skip_teaser=%s)",
                          len(self.kf_db), self.sc_skip_teaser)

            # Fallback matcher: same keyframes, sc_dist_thres disabled. The
            # DB's own matcher (self.kf_db.matcher) only ever reports the
            # accept/reject OUTCOME of Scan Context's ranking (idx=-1 below
            # sc_dist_thres) -- it has no way to hand back the best candidate
            # once rejected. This second instance exists purely so a
            # below-threshold query still resolves to Scan Context's actual
            # nearest keyframe instead of refusing to publish.
            self._sc_fallback_matcher = ScanContextMatcher(
                lidar_height=self.kf_db.sc_params["lidar_height"],
                pc_max_radius=self.kf_db.sc_params["pc_max_radius"],
                sc_dist_thres=float("inf"), num_exclude_recent=1)
            for pr in self.kf_db.prepared:
                self._sc_fallback_matcher.add_reference(np.asarray(pr.ref_cloud.points))
        else:
            # ===== unchanged: fixed-.pcd path =====
            if not self.reference_pcd_path:
                rospy.logfatal("~reference_pcd is not set. Define it in "
                               "config/params.yaml (or pass "
                               "reference_pcd:=/path/to/map.pcd). Refusing to "
                               "guess a reference map.")
                rospy.signal_shutdown("no reference")
                return
            if not os.path.isfile(self.reference_pcd_path):
                rospy.logfatal("Reference PCD not found: %s", self.reference_pcd_path)
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
            # NOTE: the working radii depend on BOTH clouds (they are derived from
            # the scene scale), so PreparedReference only caches the raw reference;
            # the downsampling/FPFH happens in align_target once the target is known.
            rospy.loginfo("Reference cached: %d raw pts",
                          len(self.prepared_ref.ref_raw))

            # Scene-scale sanity: the voxel MUST track the size of the scene
            # (~extent/200 -- a 200m outdoor map wants ~0.75-1.0m, a 17m indoor
            # scene wants ~0.10m). Too coarse washes the geometry out; too fine
            # explodes memory. Warn rather than fail, since the target's extent
            # is not known yet and the reference alone is only an estimate.
            _rp = self.prepared_ref.ref_raw
            if len(_rp):
                _extent = float((_rp[:, :2].max(axis=0) - _rp[:, :2].min(axis=0)).max())
                _suggest = _extent / 200.0
                if _suggest > 0 and not (_suggest / 3.0 <= self.params.GLOBAL_VOXEL
                                         <= _suggest * 3.0):
                    rospy.logwarn("GLOBAL_VOXEL=%.3f looks mismatched to the "
                                  "reference scene (XY extent %.1fm -> suggest "
                                  "~%.3f). Wrong scale is the #1 cause of a "
                                  "garbage transform with near-zero coverage.",
                                  self.params.GLOBAL_VOXEL, _extent, _suggest)

        # persistent FAISS GPU resources (reused, not per-call). Falls back to
        # CPU inside _correspondences if no GPU / faiss-cpu build.
        try:
            self.faiss_res = faiss.StandardGpuResources()
        except Exception:
            self.faiss_res = None

        # -------- state --------
        self.frames = []
        self.done = False
        self.target_frame_id = None

        # -------- publishers --------
        self.tf_pub = rospy.Publisher(
            self.output_topic, TransformStamped, queue_size=1, latch=True)
        self.mat_pub = rospy.Publisher(
            self.matrix_topic, Float64MultiArray, queue_size=1, latch=True)

        # static TF broadcaster
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # -------- subscriber --------
        self.sub = rospy.Subscriber(
            self.input_topic, PointCloud2, self.cloud_cb, queue_size=50)

        # -------- Scan Context live sync (only when enabled) --------
        # Buffers a synced (stamp, pose, points) stream from sc_cloud_topic /
        # sc_odom_topic, mirroring sc_keyframes.read_synced_scans's
        # nearest-timestamp pairing but incrementally, live, off the wire.
        # _sc_odom_cb and _sc_cloud_cb run on separate rospy callback threads
        # and both touch these buffers -- guard with a lock (unlike
        # self.frames above, which only one callback ever touches).
        self._sc_lock = threading.Lock()
        self.sc_scans = []
        self._sc_odom_stamps = []
        self._sc_odom_mats = []
        if self.use_scan_context:
            from nav_msgs.msg import Odometry
            self.sc_sub_odom = rospy.Subscriber(
                self.sc_odom_topic, Odometry, self._sc_odom_cb, queue_size=200)
            self.sc_sub_cloud = rospy.Subscriber(
                self.sc_cloud_topic, PointCloud2, self._sc_cloud_cb, queue_size=50)
            rospy.loginfo("Also listening on %s / %s for Scan Context retrieval.",
                          self.sc_cloud_topic, self.sc_odom_topic)

        rospy.loginfo("Listening on %s — need %d frames.",
                      self.input_topic, self.num_frames)

    def _idle(self):
        """Unregister every subscriber -- called on every terminal path
        (no-match, gate refusal, align failure, or a successful publish)."""
        self.sub.unregister()
        if self.use_scan_context:
            self.sc_sub_cloud.unregister()
            self.sc_sub_odom.unregister()

    def _sc_odom_cb(self, msg):
        if self.done:
            return
        stamp = msg.header.stamp.to_sec()
        T = self._sck.odom_to_matrix(msg)
        with self._sc_lock:
            self._sc_odom_stamps.append(stamp)
            self._sc_odom_mats.append(T)
            cutoff = stamp - 5.0   # a few seconds of history is plenty
            while self._sc_odom_stamps and self._sc_odom_stamps[0] < cutoff:
                self._sc_odom_stamps.pop(0)
                self._sc_odom_mats.pop(0)

    def _sc_cloud_cb(self, msg):
        if self.done:
            return
        stamp = msg.header.stamp.to_sec()
        pts = self._sck.pointcloud2_to_xyz(msg)
        if pts.shape[0] == 0:
            return
        with self._sc_lock:
            if not self._sc_odom_stamps:
                return
            i = bisect.bisect_left(self._sc_odom_stamps, stamp)
            best = None
            for j in (i - 1, i):
                if 0 <= j < len(self._sc_odom_stamps):
                    gap = abs(self._sc_odom_stamps[j] - stamp)
                    if best is None or gap < best[0]:
                        best = (gap, j)
            if best is None or best[0] > self.sc_max_odom_gap:
                return   # no odometry close enough in time -- drop, like a bag read would
            self.sc_scans.append(
                self._sck.SyncedScan(stamp, self._sc_odom_mats[best[1]], pts))
            accumulate_sec = self.kf_db.sc_params["accumulate_sec"]
            cutoff = stamp - max(3.0 * accumulate_sec, 2.0)
            while self.sc_scans and self.sc_scans[0].stamp < cutoff:
                self.sc_scans.pop(0)

    def cloud_cb(self, msg):
        if self.done:
            return
        if self.target_frame_id is None:
            self.target_frame_id = (self.target_frame_override
                                    or msg.header.frame_id or "target")
        cloud = pointcloud2_to_o3d(msg)
        if len(cloud.points) == 0:
            rospy.logwarn_throttle(2.0, "empty cloud frame skipped")
            return
        self.frames.append(cloud)
        rospy.loginfo_throttle(
            1.0, "accumulated %d/%d frames", len(self.frames), self.num_frames)

        if len(self.frames) >= self.num_frames:
            self.run_alignment(msg.header.stamp)

    def run_alignment(self, stamp):
        self.done = True   # set first so no re-entrancy from queued callbacks
        rospy.loginfo("Accumulated %d frames. Aligning...", len(self.frames))
        target_cloud = accumulate_clouds(self.frames)
        rospy.loginfo("Merged target points: %d", len(target_cloud.points))

        # -------- Scan Context reference selection (only when enabled) --------
        # Baseline (use_scan_context=False) path is untouched below: index/
        # yaw_rad stay unused and align_target runs against self.prepared_ref
        # exactly as before.
        index, yaw_rad = None, None
        if self.use_scan_context:
            with self._sc_lock:
                sc_scans_snapshot = list(self.sc_scans)
            if not sc_scans_snapshot:
                rospy.logerr("Scan Context: no synced %s/%s frames received; "
                             "not publishing.", self.sc_cloud_topic, self.sc_odom_topic)
                self._idle()
                return
            sp = self.kf_db.sc_params
            # same "q" the /cloud_registered_body stamp verify_pipeline.py's
            # findings.md table is keyed on -- cross-reference this value
            # against that table's rows to check whether this live query
            # landed on a no_match instant there too.
            rospy.loginfo("Scan Context: query stamp (q) = %.2f",
                          sc_scans_snapshot[-1].stamp)
            try:
                qpts, _pose, _meta = self._sck.build_sc_cloud(
                    sc_scans_snapshot, ref_pose=sc_scans_snapshot[-1].pose,
                    accumulate_sec=sp["accumulate_sec"], voxel=sp["voxel"],
                    ceiling_crop_m=sp["ceiling_crop_m"],
                    max_translation_m=sp.get("max_translation_m", 1.0))
            except ValueError as e:
                rospy.logerr("Scan Context: query window not a single "
                             "viewpoint (%s); not publishing.", e)
                self._idle()
                return
            index, yaw_rad = self.kf_db.matcher.query(qpts)
            if index < 0:
                # below sc_dist_thres -- not refusing to publish anymore;
                # fall back to Scan Context's best (nearest) candidate.
                index, yaw_rad = self._sc_fallback_matcher.query(qpts)
                rospy.logwarn("Scan Context: below confidence threshold; "
                              "using best candidate anyway -> keyframe %d, "
                              "yaw=%.1f deg (skip_teaser=%s)",
                              index, np.degrees(yaw_rad), self.sc_skip_teaser)
            else:
                rospy.loginfo("Scan Context: confident match -> keyframe %d, "
                              "yaw=%.1f deg (skip_teaser=%s)",
                              index, np.degrees(yaw_rad), self.sc_skip_teaser)

        try:
            if self.use_scan_context:
                T_full, info = self._relocalize(
                    target_cloud, index, yaw_rad, self.kf_db,
                    params=self.params, faiss_res=self.faiss_res,
                    logger=lambda m: rospy.loginfo("  [align] %s", m),
                    skip_teaser=self.sc_skip_teaser)
            else:
                T_full, info = align_target(
                    target_cloud, self.prepared_ref, self.params,
                    faiss_res=self.faiss_res,
                    logger=lambda m: rospy.loginfo("  [align] %s", m))
        except Exception as e:
            rospy.logerr("Alignment failed: %s", str(e))
            rospy.logerr("Not publishing. Shutting down.")
            rospy.signal_shutdown("alignment failed")
            return

        rospy.loginfo("=== ALIGNMENT RESULT ===")
        cov  = info.get("coverage", 0.0)
        rmse = info.get("inlier_rmse", float("inf"))
        # teaser_yaw_deg is None on the skip_teaser=True (SC-seeded) path --
        # TEASER never ran, so there is no TEASER yaw to report.
        teaser_yaw = info.get("teaser_yaw_deg")
        teaser_yaw_str = "%.2fdeg" % teaser_yaw if teaser_yaw is not None else "N/A (skip_teaser)"
        rospy.loginfo("teaser_yaw=%s  final_yaw=%.2fdeg  tilt=%.2fdeg  "
                      "inliers=%d/%d  coverage=%.2f  rmse=%.2fm  t=[%.2f %.2f %.2f]",
                      teaser_yaw_str, info.get("yaw_deg", 0.0),
                      info.get("tilt_deg", 0.0),
                      info.get("teaser_inliers", -1),
                      info.get("n_correspondences", -1),
                      cov, rmse, info.get("tx", 0.0), info.get("ty", 0.0),
                      info.get("tz", 0.0))
        np.set_printoptions(precision=4, suppress=True)
        rospy.loginfo("T (target -> reference):\n%s", str(T_full))

        if (not np.isfinite(rmse)) or (cov < self.min_coverage) or \
           (rmse > self.max_rmse_m):
            rospy.logerr("REFUSING to publish: coverage=%.2f (min %.2f) "
                         "rmse=%.2fm (max %.2fm).",
                         cov, self.min_coverage, rmse, self.max_rmse_m)
            rospy.loginfo("Done (no publish). Node will idle; Ctrl-C to exit.")
            self._idle()
            return

        # publish TransformStamped
        ts = matrix_to_transform_stamped(
            T_full, self.reference_frame, self.target_frame_id, stamp)
        self.tf_pub.publish(ts)

        # publish raw 4x4 (row-major) as Float64MultiArray
        mat = Float64MultiArray()
        mat.layout.dim = [
            MultiArrayDimension(label="rows", size=4, stride=16),
            MultiArrayDimension(label="cols", size=4, stride=4),
        ]
        mat.data = [float(x) for x in T_full.flatten()]
        self.mat_pub.publish(mat)

        # Broadcast reference_frame -> odom_parent_frame (= T_full). TF then
        # auto-composes with FAST-LIO's odom_parent_frame -> body, so any
        # consumer can look up body-in-reference without republishing odometry.
        if self.broadcast_tf:
            static_ts = matrix_to_transform_stamped(
                T_full, self.reference_frame, self.odom_parent_frame, stamp)
            self.static_tf_broadcaster.sendTransform(static_ts)
            rospy.loginfo("Broadcast static TF: %s -> %s",
                          self.reference_frame, self.odom_parent_frame)

        rospy.loginfo("Published transform on %s and matrix on %s (latched).",
                      self.output_topic, self.matrix_topic)
        rospy.loginfo("Done. Node will idle; Ctrl-C to exit.")
        self._idle()


if __name__ == "__main__":
    try:
        AlignmentNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass