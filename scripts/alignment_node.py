#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# When launched via catkin's devel-space wrapper, this script is exec'd from a
# different directory, so its sibling modules (alignment_core, ros_cloud_utils)
# are not on sys.path. Add this file's real directory so the imports resolve
# regardless of how the node is started.
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

        # ---- parameters ----
        self.reference_pcd_path = os.path.expanduser(rospy.get_param(
            "~reference_pcd", "~/maps/reference_prepared.pcd"))
        self.input_topic = rospy.get_param("~input_topic", "/cloud_registered")
        self.output_topic = rospy.get_param("~output_topic", "/target_to_reference")
        self.matrix_topic = rospy.get_param("~matrix_topic", "/target_to_reference_matrix")
        self.num_frames = int(rospy.get_param("~num_frames", 10))
        self.reference_frame = rospy.get_param("~reference_frame", "reference_map")
        # target frame: read from incoming msg header, or override
        self.target_frame_override = rospy.get_param("~target_frame", "")
        # ---- static TF (Option 1): broadcast reference_frame -> odom_parent_frame
        # so FAST-LIO's odom_parent -> body chain composes into the reference frame.
        self.broadcast_tf = bool(rospy.get_param("~broadcast_tf", True))
        # FAST-LIO's odometry parent frame (what its camera_init/odom is called).
        self.odom_parent_frame = rospy.get_param("~odom_parent_frame", "camera_init")

        # ---- parameters object (allow overrides from param server) ----
        self.params = AlignParams()
        for name in ("RANSAC_VOXEL", "ICP_VOXEL", "GROUND_REMOVE_Z",
                     "TEASER_NOISE_BOUND", "ICP_DIST_COARSE", "ICP_DIST_FINE"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, float(pv))
                rospy.loginfo("param override %s=%s", name, pv)

        # ---- load + prepare reference (once) ----
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
        rospy.loginfo("Reference points: %d. Preprocessing (once)...",
                      len(ref_cloud.points))
        self.prepared_ref = PreparedReference(ref_cloud, self.params)
        rospy.loginfo("Reference prepared. Down pts: %d",
                      len(self.prepared_ref.ref_down.points))

        # ---- persistent FAISS GPU resources (reused, not per-call) ----
        self.faiss_res = faiss.StandardGpuResources()

        # ---- state ----
        self.frames = []
        self.done = False
        self.target_frame_id = None

        # ---- publishers (latched so late subscribers get the one-shot result) ----
        self.tf_pub = rospy.Publisher(
            self.output_topic, TransformStamped, queue_size=1, latch=True)
        self.mat_pub = rospy.Publisher(
            self.matrix_topic, Float64MultiArray, queue_size=1, latch=True)

        # ---- static TF broadcaster (latched by design; persists for the session)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # ---- subscriber ----
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
        rospy.loginfo("TEASER fitness=%.4f  GICP fitness=%.4f  GICP rmse=%.3fcm",
                      info.get("teaser_fitness", -1),
                      info.get("icp_fitness", -1),
                      info.get("icp_rmse", -1) * 100)
        np.set_printoptions(precision=4, suppress=True)
        rospy.loginfo("T (target -> reference):\n%s", str(T_full))

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

        # ---- Option 1: static TF so FAST-LIO odometry composes into reference ----
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