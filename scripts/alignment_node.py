#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from collections import deque

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Vector3, Vector3Stamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from tf.transformations import quaternion_matrix

import open3d as o3d
import faiss

from alignment_core import AlignParams, PreparedReference, align_target
from ros_cloud_utils import (pointcloud2_to_o3d, accumulate_clouds,
                            matrix_to_transform_stamped)


# =============================================================================
# Gravity tracker — extracts the gravity vector from whatever the bag publishes
# so levelling is deterministic (no RANSAC plane fit). Priority:
#   1. explicit gravity topic (geometry_msgs/Vector3[Stamped])  -- best
#   2. FAST-LIO /Odometry orientation
#   3. raw IMU linear acceleration                              -- fallback
# The vector is reported in the CLOUD frame, pointing DOWN.
# =============================================================================
class GravityTracker(object):
    def __init__(self, grav_topic="", odom_topic="/Odometry",
                 imu_topic="/livox/imu", cloud_is_world=True, imu_avg_n=100):
        self._g = None                        # (unit_vector_down, source)
        self._imu_buf = deque(maxlen=imu_avg_n)
        self._cloud_is_world = cloud_is_world
        self._subs = []

        if grav_topic:
            # subscribe to both stamped and plain; whichever type the bag has fires
            self._subs.append(rospy.Subscriber(
                grav_topic, Vector3Stamped, self._cb_grav_stamped, queue_size=50))
            self._subs.append(rospy.Subscriber(
                grav_topic, Vector3, self._cb_grav, queue_size=50))
        if odom_topic:
            self._subs.append(rospy.Subscriber(
                odom_topic, Odometry, self._cb_odom, queue_size=50))
        if imu_topic:
            self._subs.append(rospy.Subscriber(
                imu_topic, Imu, self._cb_imu, queue_size=200))

    def _set(self, g, source):
        g = np.asarray(g, dtype=np.float64)
        n = np.linalg.norm(g)
        if n < 1e-6:
            return
        self._g = (g / n, source)

    def _cb_grav_stamped(self, msg):
        v = msg.vector
        self._set([v.x, v.y, v.z], "grav_topic")

    def _cb_grav(self, msg):
        self._set([msg.x, msg.y, msg.z], "grav_topic")

    def _cb_odom(self, msg):
        # World-frame down is [0,0,-1]. If the cloud we level is in the world
        # (camera_init) frame, gravity there is just [0,0,-1]; otherwise rotate
        # world-down into the body frame using the odom orientation.
        q = msg.pose.pose.orientation
        R_wb = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]  # body -> world
        down_world = np.array([0.0, 0.0, -1.0])
        g = down_world if self._cloud_is_world else (R_wb.T @ down_world)
        # don't clobber an explicit gravity topic if one is publishing
        if self._g is None or self._g[1] != "grav_topic":
            self._set(g, "odom")

    def _cb_imu(self, msg):
        a = msg.linear_acceleration
        self._imu_buf.append([a.x, a.y, a.z])
        # gravity ~= mean specific force when near-static; IMU reads +g when
        # level, so DOWN = -mean(acc). Only use IMU if nothing better exists.
        if (self._g is None or self._g[1] == "imu") and len(self._imu_buf) >= 5:
            self._set(-np.mean(self._imu_buf, axis=0), "imu")

    def get(self):
        """Return (gravity_unit_down, source) or (None, None). Call at the END
        of the accumulation window, when FAST-LIO's estimate has converged."""
        if self._g is None:
            return None, None
        return self._g[0].copy(), self._g[1]


class AlignmentNode(object):
    def __init__(self):
        rospy.init_node("cloud_alignment_node")

        # parameters 
        self.reference_pcd_path = os.path.expanduser(rospy.get_param(
            "~reference_pcd", "~/maps/reference_jul1.pcd"))
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

        # gravity-levelling parameters 
        # Which topics carry gravity info in the bag. Leave gravity_topic empty
        # unless you added an explicit publisher to FAST-LIO. odom_topic is the
        # normal source; imu_topic is the fallback.
        self.gravity_topic = rospy.get_param("~gravity_topic", "")
        self.odom_topic    = rospy.get_param("~odom_topic", "/Odometry")
        self.imu_topic     = rospy.get_param("~imu_topic", "/livox/imu")
        # /cloud_registered is in the camera_init (world) frame in stock FAST-LIO,
        # which is gravity-aligned at init -> gravity there is ~[0,0,-1].
        self.cloud_is_world = bool(rospy.get_param("~cloud_is_world", True))
        # Master switch: if False, ignore gravity and use the RANSAC plane fit.
        self.use_gravity = bool(rospy.get_param("~use_gravity_levelling", True))

        # parameters object (allow overrides from param server) 
        self.params = AlignParams()
        for name in ("RANSAC_VOXEL", "ICP_VOXEL", "GROUND_REMOVE_Z",
                     "TEASER_NOISE_BOUND", "ICP_DIST_COARSE", "ICP_DIST_FINE"):
            pv = rospy.get_param("~%s" % name, None)
            if pv is not None:
                setattr(self.params, name, float(pv))
                rospy.loginfo("param override %s=%s", name, pv)
        self.params.USE_GRAVITY_LEVELLING = self.use_gravity

        # gravity tracker (subscribes immediately so it's warm before alignment) 
        self.gravity_tracker = None
        if self.use_gravity:
            self.gravity_tracker = GravityTracker(
                grav_topic=self.gravity_topic,
                odom_topic=self.odom_topic,
                imu_topic=self.imu_topic,
                cloud_is_world=self.cloud_is_world)
            rospy.loginfo(
                "Gravity levelling ON (grav_topic=%r odom=%r imu=%r world=%s)",
                self.gravity_topic, self.odom_topic, self.imu_topic,
                self.cloud_is_world)
        else:
            rospy.loginfo("Gravity levelling OFF — using RANSAC plane fit.")

        # load + prepare reference (once) 
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

        # Reference gravity: the reference PCD was captured in its OWN camera_init
        # frame. If it's world-aligned (cloud_is_world), gravity is [0,0,-1] and
        # we can supply that directly for a deterministic reference levelling
        # without needing a live message. If you instead prepared the reference
        # from a bag and want its true gravity, set ~reference_gravity as a list.
        ref_gravity = None
        if self.use_gravity:
            rg = rospy.get_param("~reference_gravity", None)  # e.g. [0,0,-1]
            if rg is not None and len(rg) == 3:
                ref_gravity = np.asarray(rg, dtype=np.float64)
            elif self.cloud_is_world:
                ref_gravity = np.array([0.0, 0.0, -1.0])
            rospy.loginfo("Reference gravity for levelling: %s",
                          str(ref_gravity))

        self.prepared_ref = PreparedReference(
            ref_cloud, self.params, gravity=ref_gravity)
        rospy.loginfo("Reference prepared. Down pts: %d",
                      len(self.prepared_ref.ref_down.points))

        # persistent FAISS GPU resources (reused, not per-call) 
        self.faiss_res = faiss.StandardGpuResources()

        # state 
        self.frames = []
        self.done = False
        self.target_frame_id = None

        # publishers 
        self.tf_pub = rospy.Publisher(
            self.output_topic, TransformStamped, queue_size=1, latch=True)
        self.mat_pub = rospy.Publisher(
            self.matrix_topic, Float64MultiArray, queue_size=1, latch=True)

        # static TF broadcaster
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # subscriber 
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

        # Sample gravity at the END of the accumulation window (best-converged
        # FAST-LIO estimate). None -> align_target uses the plane-fit fallback.
        gravity = None
        if self.gravity_tracker is not None:
            gravity, gsrc = self.gravity_tracker.get()
            if gravity is not None:
                rospy.loginfo("Levelling from gravity (%s): %s",
                              gsrc, np.round(gravity, 4))
            else:
                rospy.logwarn("No gravity message received yet — "
                              "falling back to RANSAC plane fit.")

        try:
            T_full, info = align_target(
                target_cloud, self.prepared_ref, self.params,
                faiss_res=self.faiss_res, gravity=gravity,
                logger=lambda m: rospy.loginfo("  [align] %s", m))
        except Exception as e:
            rospy.logerr("Alignment failed: %s", str(e))
            rospy.logerr("Not publishing. Shutting down.")
            rospy.signal_shutdown("alignment failed")
            return

        rospy.loginfo("=== ALIGNMENT RESULT ===")
        rospy.loginfo("levelling=%s  TEASER fitness=%.4f  GICP fitness=%.4f  "
                      "GICP rmse=%.3fcm",
                      "gravity" if info.get("used_gravity_levelling") else "plane",
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