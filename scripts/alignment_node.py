#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import time

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


class AlignmentNode(object):
    def __init__(self):
        rospy.init_node("cloud_alignment_node")

        # -------- I/O parameters --------
        self.reference_pcd_path = os.path.expanduser(
            rospy.get_param("~reference_pcd", ""))
        self.input_topic = rospy.get_param("~input_topic", "/cloud_registered")
        self.output_topic = rospy.get_param("~output_topic", "/target_to_reference")
        self.matrix_topic = rospy.get_param("~matrix_topic", "/target_to_reference_matrix")
        self.num_frames = int(rospy.get_param("~num_frames", 120))
        self.reference_frame = rospy.get_param("~reference_frame", "reference_map")
        # target frame: read from incoming msg header, or override
        self.target_frame_override = rospy.get_param("~target_frame", "")
        # static TF : broadcast reference_frame -> odom_parent_frame
        # so FAST-LIO's odom_parent -> body chain composes into the reference frame.
        self.broadcast_tf = bool(rospy.get_param("~broadcast_tf", True))
        # FAST-LIO's odometry parent frame (what its camera_init/odom is called).
        self.odom_parent_frame = rospy.get_param("~odom_parent_frame", "camera_init")

        # Publish gate thresholds (optional; both default OFF) check params.yaml
        self.min_coverage = float(rospy.get_param("~min_coverage", 0.0))
        self.max_rmse_m = float(rospy.get_param("~max_rmse_m", 1e9))

        # Alignment parameters (global map-to-map registration) 
        self.params = AlignParams()

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

        # multi-scale sweep (alignment_core._sweep) 
        pv = rospy.get_param("~GLOBAL_SWEEP", None)
        if pv is not None:
            self.params.GLOBAL_SWEEP = bool(pv)
        pv = rospy.get_param("~GLOBAL_VOXELS", None)
        if pv is not None:
            self.params.GLOBAL_VOXELS = [float(x) for x in pv] if pv else None
        pv = rospy.get_param("~GLOBAL_AUTO_FACTORS", None)
        if pv:
            self.params.GLOBAL_AUTO_FACTORS = tuple(float(x) for x in pv)
        for name in ("CONSENSUS_ROT_DEG", "CONSENSUS_TRANS_FACTOR"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, float(pv))
        for name in ("GLOBAL_TARGET_POINTS", "GLOBAL_MAX_POINTS",
                     "CONSENSUS_MIN_AGREE"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, int(pv))
        if rospy.has_param("~GLOBAL_MAX_TILT_DEG"):
            pv = rospy.get_param("~GLOBAL_MAX_TILT_DEG")
            self.params.GLOBAL_MAX_TILT_DEG = None if pv is None else float(pv)
        # publish gate: refuse to publish when the scales did not agree
        self.require_consensus = bool(rospy.get_param("~require_consensus", True))

        if self.params.GLOBAL_SWEEP:
            rospy.loginfo("global-reg: SWEEP mode  voxels=%s  target_points=%d "
                          "factors=%s  consensus: rot<%.1fdeg trans<%.1fxvoxel "
                          "min_agree=%d  max_tilt=%s  require_consensus=%s",
                          self.params.GLOBAL_VOXELS or "auto-from-target",
                          self.params.GLOBAL_TARGET_POINTS,
                          list(self.params.GLOBAL_AUTO_FACTORS),
                          self.params.CONSENSUS_ROT_DEG,
                          self.params.CONSENSUS_TRANS_FACTOR,
                          self.params.CONSENSUS_MIN_AGREE,
                          self.params.GLOBAL_MAX_TILT_DEG,
                          self.require_consensus)
        else:
            # Fixed mode: log the final scale set (see the incoherence check below).
            rospy.loginfo("global-reg: FIXED mode  voxel=%.3f normal_r=%.3f "
                          "feat_r=%.3f teaser_nb=%.3f", self.params.GLOBAL_VOXEL,
                          self.params.GLOBAL_NORMAL_R,
                          self.params.GLOBAL_FEATURE_R,
                          self.params.GLOBAL_TEASER_NB)
        # Fixed mode only: refuse to fail silently on an incoherent scale set. A
        # normal/feature radius at or below the voxel pitch means each
        # descriptor sees ~1 neighbour -> meaningless FPFH -> an arbitrary
        # transform with ~0 coverage (see the comment above).
        if not self.params.GLOBAL_SWEEP and (
                self.params.GLOBAL_NORMAL_R <= self.params.GLOBAL_VOXEL or
                self.params.GLOBAL_FEATURE_R <= self.params.GLOBAL_VOXEL):
            rospy.logerr("INCOHERENT global-reg scale: normal_r=%.3f / "
                         "feat_r=%.3f are not larger than voxel=%.3f. FPFH will "
                         "be computed from ~1 neighbour and registration WILL "
                         "fail (arbitrary rotation, coverage ~0). Leave the "
                         "GLOBAL_NORMAL_R/GLOBAL_FEATURE_R/GLOBAL_TEASER_NB "
                         "params unset so they derive from ~GLOBAL_VOXEL.",
                         self.params.GLOBAL_NORMAL_R,
                         self.params.GLOBAL_FEATURE_R, self.params.GLOBAL_VOXEL)

        # load reference (once) 
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
        rospy.loginfo("Reference cached: %d raw pts",
                      len(self.prepared_ref.ref_raw))

        _rp = self.prepared_ref.ref_raw
        if len(_rp) and not self.params.GLOBAL_SWEEP:
            _extent = float((_rp[:, :2].max(axis=0) - _rp[:, :2].min(axis=0)).max())
            _suggest = _extent / 200.0
            if _suggest > 0 and not (_suggest / 3.0 <= self.params.GLOBAL_VOXEL
                                     <= _suggest * 3.0):
                rospy.logwarn("GLOBAL_VOXEL=%.3f looks mismatched to the "
                              "reference scene (XY extent %.1fm -> suggest "
                              "~%.3f). Wrong scale is the #1 cause of a "
                              "garbage transform with near-zero coverage.",
                              self.params.GLOBAL_VOXEL, _extent, _suggest)

        # persistent FAISS GPU resources (reused, not per-call). Falls back to CPU inside _correspondences if no GPU / faiss-cpu build.
        try:
            self.faiss_res = faiss.StandardGpuResources()
        except Exception:
            self.faiss_res = None

        # state 
        self.frames = []
        self.done = False
        self.target_frame_id = None
        # accumulation bookkeeping (cloud_cb and the stall timer run on
        # different threads -> _acc_lock makes "align now" happen exactly once)
        self._acc_lock = threading.Lock()
        self._last_msg_stamp = None   # rospy.Time of the newest frame, for TF
        self._last_rx_wall = None     # time.monotonic() when the newest arrived

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

        # End-of-stream watchdog: if the stream stops before num_frames, align with what was collected.
        self._watchdog = threading.Thread(target=self._stall_watchdog)
        self._watchdog.daemon = True
        self._watchdog.start()

        rospy.loginfo("Listening on %s -- need %d frames.",
                      self.input_topic, self.num_frames)

    def _idle(self):
        """Unregister the input subscriber -- called on the terminal paths
        (gate refusal or a successful publish)."""
        self.sub.unregister()

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
        with self._acc_lock:
            if self.done:
                return
            self.frames.append(cloud)
            self._last_msg_stamp = msg.header.stamp
            self._last_rx_wall = time.monotonic()
            n = len(self.frames)
            rospy.loginfo_throttle(1.0, "accumulated %d/%d frames",
                                   n, self.num_frames)
            if n < self.num_frames:
                return
            frames, stamp = self._claim_frames()
        self.run_alignment(frames, stamp)

    def _claim_frames(self):
        """Mark accumulation finished and hand back the frames. CALLER MUST
        HOLD _acc_lock -- the single point that decides 'align now'."""
        self.done = True
        frames, self.frames = self.frames, []
        return frames, self._last_msg_stamp

    def _stall_watchdog(self):
        """If frames have started but none arrived for STREAM_END_SEC (wall
        time) before num_frames was reached, the stream has ended: align with
        the frames collected so far."""
        STREAM_END_SEC = 3.0
        while not rospy.is_shutdown():
            time.sleep(0.5)
            with self._acc_lock:
                if self.done:
                    return
                if (self._last_rx_wall is None or time.monotonic()
                        - self._last_rx_wall < STREAM_END_SEC):
                    continue
                rospy.logwarn("Stream ended after %d/%d frames. Aligning with "
                              "what we have.", len(self.frames), self.num_frames)
                frames, stamp = self._claim_frames()
            self.run_alignment(frames, stamp)
            return

    def run_alignment(self, frames, stamp):
        rospy.loginfo("Accumulated %d frames. Aligning...", len(frames))
        target_cloud = accumulate_clouds(frames)
        rospy.loginfo("Merged target points: %d", len(target_cloud.points))

        try:
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
        teaser_yaw = info.get("teaser_yaw_deg")
        teaser_yaw_str = "%.2fdeg" % teaser_yaw if teaser_yaw is not None else "N/A"
        rospy.loginfo("teaser_yaw=%s  final_yaw=%.2fdeg  tilt=%.2fdeg  "
                      "inliers=%d/%d  coverage=%.2f  rmse=%.2fm  t=[%.2f %.2f %.2f]",
                      teaser_yaw_str, info.get("yaw_deg", 0.0),
                      info.get("tilt_deg", 0.0),
                      info.get("teaser_inliers", -1),
                      info.get("n_correspondences", -1),
                      cov, rmse, info.get("tx", 0.0), info.get("ty", 0.0),
                      info.get("tz", 0.0))
        if info.get("path") == "sweep":
            rospy.loginfo("sweep: chose voxel=%.3f  consensus=%s (%d/%d scales "
                          "agree)  tried=%s", info["voxel"], info["consensus"],
                          info["n_agree"], info["n_scales"],
                          ["%.3f" % v for v in info["voxels_tried"]])
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
        # info has no "consensus" key outside sweep mode -> gate not applied
        if self.require_consensus and info.get("consensus") is False:
            rospy.logerr("REFUSING to publish: only %d/%d scales agreed (need "
                         "%d). The scales disagree, so no single answer can be "
                         "trusted. Check overlap, or set require_consensus: "
                         "false to publish anyway.", info.get("n_agree", 0),
                         info.get("n_scales", 0),
                         self.params.CONSENSUS_MIN_AGREE)
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
