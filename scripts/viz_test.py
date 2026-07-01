#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_live_result.py
------------------------
Checks the transform that the LIVE ROS node published.

Subscribes to /target_to_reference_matrix (latched), grabs the 4x4, applies it
to a target PCD, and overlays it on the reference PCD:
  blue   = reference
  orange = target transformed by the node's published matrix

This verifies the actual deployed result, not an offline recomputation.

Run AFTER the node has published (the topic is latched, so timing is flexible):
  rosrun cloud_aligner visualize_live_result.py \
      _reference_pcd:=$HOME/catkin_ws/maps/reference1.pcd \
      _target_pcd:=$HOME/catkin_ws/maps/test_tgt.pcd

Note: the node aligned the first N live frames, which differ from a whole-bag
target PCD. For the closest visual match, point _target_pcd at a cloud built
from the same first frames (see --help in the node). Using the full target PCD
still shows whether the rotation/translation is sane.
"""

import os
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import open3d as o3d


class LiveResultViz(object):
    def __init__(self):
        rospy.init_node("visualize_live_result")
        self.ref_path = os.path.expanduser(
            rospy.get_param("~reference_pcd",
                            "~/catkin_ws/maps/reference1.pcd"))
        self.tgt_path = os.path.expanduser(
            rospy.get_param("~target_pcd",
                            "~/catkin_ws/maps/test_tgt.pcd"))
        self.matrix_topic = rospy.get_param("~matrix_topic",
                                            "/target_to_reference_matrix")
        self.T = None
        rospy.Subscriber(self.matrix_topic, Float64MultiArray,
                         self.cb, queue_size=1)
        rospy.loginfo("Waiting for matrix on %s (latched) ...",
                      self.matrix_topic)

    def cb(self, msg):
        if self.T is not None:
            return
        self.T = np.array(msg.data, dtype=np.float64).reshape(4, 4)
        rospy.loginfo("Got matrix:\n%s", str(self.T))

    def run(self):
        # wait up to 10s for the latched matrix
        t0 = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.T is None:
            if (rospy.Time.now() - t0).to_sec() > 10.0:
                rospy.logfatal("No matrix received. Did the node publish?")
                return
            rate.sleep()
        if self.T is None:
            return

        ref = o3d.io.read_point_cloud(self.ref_path)
        tgt = o3d.io.read_point_cloud(self.tgt_path)
        assert len(ref.points) and len(tgt.points), "empty cloud(s)"

        tgt_moved = o3d.geometry.PointCloud(tgt)
        tgt_moved.transform(self.T)

        ref_s = ref.voxel_down_sample(0.10)
        tgt_s = tgt_moved.voxel_down_sample(0.10)
        ref_s.paint_uniform_color([0.20, 0.50, 1.00])
        tgt_s.paint_uniform_color([1.00, 0.50, 0.10])

        rospy.loginfo("Viewer: blue=reference orange=target(published T). "
                      "Close window to exit.")
        o3d.visualization.draw_geometries(
            [ref_s, tgt_s],
            window_name="LIVE result: blue=ref orange=target",
            width=1280, height=800)


if __name__ == "__main__":
    try:
        LiveResultViz().run()
    except rospy.ROSInterruptException:
        pass