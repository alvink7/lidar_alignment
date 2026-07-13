#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

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
        self.reference_pcd_path = os.path.expanduser(rospy.get_param(
            "~reference_pcd", "~/catkin_ws/data/maps/droneparkscan.pcd"))
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

        # -------- Publish gate thresholds (optional; both default OFF) --------
        #   ~min_coverage : refuse to publish below this coverage
        #   ~max_rmse_m   : refuse to publish above this inlier RMSE (metres)
        self.min_coverage = float(rospy.get_param("~min_coverage", 0.0))
        self.max_rmse_m = float(rospy.get_param("~max_rmse_m", 1e9))

        # -------- Alignment parameters (FIXED scales, from the working notebook) --
        # These are the values that succeeded on the Apr4 bags; do not auto-scale
        # them. FPFH runs at RANSAC_VOXEL=0.30 / FEATURE_RADIUS_R=1.50, TEASER
        # noise 0.60. Levelling fits the ground plane over the low-Z floor points
        # and rotates the normal to +Z:
        #   ~RANSAC_VOXEL / ~FEATURE_RADIUS_R : FPFH scale
        #   ~TEASER_NOISE_BOUND               : max correspondence error (m)
        #   ~FLOOR_SEARCH_PCT / ~FLOOR_BAND_M : floor-point selection
        # Levelling is pure Z-shift (never rotates), so there is no tilt/gate knob.
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

        # -------- load + prepare reference (once) --------
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

        rospy.loginfo("Listening on %s — need %d frames.",
                      self.input_topic, self.num_frames)

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
        rospy.loginfo("level_tilt ref=%.2f tgt=%.2f deg | teaser_yaw=%.2fdeg "
                      "final_yaw=%.2fdeg  inliers=%d/%d  coverage=%.2f  "
                      "rmse=%.2fm  median=%.2fm",
                      info.get("ref_tilt_deg", -1), info.get("tgt_tilt_deg", -1),
                      info.get("teaser_yaw_deg", 0.0), info.get("yaw_deg", 0.0),
                      info.get("teaser_inliers", -1),
                      info.get("n_correspondences", -1),
                      cov, rmse, info.get("median_dist", -1))
        np.set_printoptions(precision=4, suppress=True)
        rospy.loginfo("T (target -> reference):\n%s", str(T_full))

        if (not np.isfinite(rmse)) or (cov < self.min_coverage) or \
           (rmse > self.max_rmse_m):
            rospy.logerr("REFUSING to publish: coverage=%.2f (min %.2f) "
                         "rmse=%.2fm (max %.2fm).",
                         cov, self.min_coverage, rmse, self.max_rmse_m)
            rospy.loginfo("Done (no publish). Node will idle; Ctrl-C to exit.")
            self.sub.unregister()
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
        self.sub.unregister()


if __name__ == "__main__":
    try:
        AlignmentNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass