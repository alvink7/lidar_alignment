#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_test.py
-----------
Overlay the reference and the published transform's target, live off
/target_to_reference_matrix.

WHAT CHANGED vs the original, and why
    The original could fail in three ways that all look identical from outside
    -- "it just doesn't come up" -- because each one is silent:

    1. SIM TIME WITH NO CLOCK. rospy.Time.now() returns 0 until something
       publishes /clock, so under `rosparam set use_sim_time true` with no bag
       playing, rate.sleep() blocks forever AND the 10 s timeout never fires
       (it compares two zeros). The node hangs with no output. This is easy to
       walk into because use_sim_time is sticky on the master -- set it once for
       a bag replay and it stays set for every node you start afterwards, until
       roscore restarts. Now: checked up front, and the wait uses WALL time.

    2. NO GL CONTEXT. draw_geometries() cannot create a window when there is no
       DISPLAY (headless, SSH without -X, broken WSLg) and it returns NORMALLY,
       without raising and without a window. Now: the window is created
       explicitly via Visualizer.create_window(), whose return value actually
       reports failure, and the environment is printed when it does.

    3. A DEGENERATE TRANSFORM. If the matrix arrives all-zero or with NaNs, the
       transformed cloud has no finite bounding box, the viewer cannot place a
       camera, and you get an empty or missing window rather than an error.
       Now: validated before use.

    On any window failure it writes the overlay to disk (a coloured .pcd, plus
    a PNG if offscreen rendering is available) so the result is still
    inspectable -- in CloudCompare, or on another machine.

USAGE
    rosrun cloud_aligner viz_test.py \
        _reference_pcd:=$HOME/catkin_ws/maps/reference_upscaled.pcd \
        _target_pcd:=$HOME/catkin_ws/maps/test_target.pcd

    _out_dir:=~/viz     where the fallback files go (default: cwd)
    _timeout:=10.0      seconds to wait for the latched matrix
    _voxel:=0.10        display downsample
    _no_window:=true    skip the viewer entirely, just write the files
"""

import os
import sys
import time

import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray

import open3d as o3d


class LiveResultViz(object):

    def __init__(self):
        rospy.init_node("visualize_live_result")
        self.ref_path = os.path.expanduser(
            rospy.get_param("~reference_pcd",
                            "~/catkin_ws/data/maps/droneparkscan.pcd"))
        self.tgt_path = os.path.expanduser(
            rospy.get_param("~target_pcd",
                            "~/catkin_ws/data/maps/palio2.pcd"))
        self.matrix_topic = rospy.get_param("~matrix_topic",
                                            "/target_to_reference_matrix")
        self.out_dir = os.path.expanduser(str(rospy.get_param("~out_dir", ".")))
        self.timeout = float(rospy.get_param("~timeout", 10.0))
        self.voxel = float(rospy.get_param("~voxel", 0.10))
        self.no_window = bool(rospy.get_param("~no_window", False))
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        # ---- sim-time trap ------------------------------------------------
        # Sticky on the master: whoever last ran a bag replay left it set.
        if rospy.get_param("/use_sim_time", False):
            rospy.logwarn(
                "/use_sim_time is TRUE. rospy.Time.now() stays at 0 until "
                "something publishes /clock -- if no bag is playing, any "
                "rospy.Rate or Duration here would block forever. This node "
                "waits on WALL time, so it is fine, but ANY other node you "
                "start right now will hang. Clear it with: "
                "rosparam set use_sim_time false")

        self.T = None
        rospy.Subscriber(self.matrix_topic, Float64MultiArray, self.cb,
                         queue_size=1)
        rospy.loginfo("Waiting for matrix on %s (latched) ...",
                      self.matrix_topic)

    def cb(self, msg):
        if self.T is not None:
            return
        if len(msg.data) != 16:
            rospy.logerr("matrix has %d elements, expected 16", len(msg.data))
            return
        self.T = np.array(msg.data, dtype=np.float64).reshape(4, 4)
        rospy.loginfo("Got matrix:\n%s", str(self.T))

    # ------------------------------------------------------------------ main
    def run(self):
        if not self._wait_for_matrix():
            return
        if not self._validate_T():
            return

        ref = o3d.io.read_point_cloud(self.ref_path)
        tgt = o3d.io.read_point_cloud(self.tgt_path)
        if not len(ref.points) or not len(tgt.points):
            rospy.logfatal("empty cloud(s): ref=%d (%s) tgt=%d (%s)",
                           len(ref.points), self.ref_path,
                           len(tgt.points), self.tgt_path)
            return
        rospy.loginfo("ref %d pts, tgt %d pts", len(ref.points), len(tgt.points))

        tgt_moved = o3d.geometry.PointCloud(tgt)
        tgt_moved.transform(self.T)

        ref_s = ref.voxel_down_sample(self.voxel)
        tgt_s = tgt_moved.voxel_down_sample(self.voxel)
        ref_s.paint_uniform_color([0.20, 0.50, 1.00])   # blue
        tgt_s.paint_uniform_color([1.00, 0.50, 0.10])   # orange

        # A quick sanity number while we are here: if the overlay looks wrong,
        # this usually already says so.
        try:
            d = np.asarray(tgt_s.compute_point_cloud_distance(ref_s))
            rospy.loginfo("target->reference NN distance: median %.3f m, "
                          "%.1f%% within 0.30 m",
                          float(np.median(d)), 100.0 * float((d < 0.30).mean()))
        except Exception:
            pass

        if self.no_window or not self._show([ref_s, tgt_s]):
            self._fallback(ref_s, tgt_s)

    # --------------------------------------------------------------- helpers
    def _wait_for_matrix(self):
        """WALL-clock wait. rospy.Time.now() is unusable here: under sim time
        with no /clock it never advances, so the timeout never fires and
        rospy.Rate.sleep() blocks indefinitely."""
        t0 = time.time()
        while not rospy.is_shutdown() and self.T is None:
            if time.time() - t0 > self.timeout:
                rospy.logfatal(
                    "No matrix on %s after %.0fs. The publisher latches, so a "
                    "running aligner would have delivered it immediately. "
                    "Check:  rostopic echo -n1 %s   and   rostopic info %s",
                    self.matrix_topic, self.timeout, self.matrix_topic,
                    self.matrix_topic)
                return False
            time.sleep(0.05)
        return self.T is not None

    def _validate_T(self):
        """An all-zero or NaN matrix produces a cloud with no finite bounding
        box; the viewer then cannot place a camera and shows nothing, with no
        error. Catch it here where it can be reported."""
        T = self.T
        if not np.all(np.isfinite(T)):
            rospy.logfatal("matrix contains non-finite values:\n%s", str(T))
            return False
        if np.allclose(T, 0.0):
            rospy.logfatal("matrix is all zeros -- the publisher sent an "
                           "uninitialised message")
            return False
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
            rospy.logwarn("bottom row is %s, not [0 0 0 1] -- is the data "
                          "row-major?", str(T[3]))
        det = float(np.linalg.det(T[:3, :3]))
        if abs(det - 1.0) > 1e-3:
            rospy.logwarn("rotation block det=%.6f (expected 1.0) -- the "
                          "matrix may be transposed or scaled", det)
        return True

    def _show(self, geoms):
        """Create the window EXPLICITLY. draw_geometries() swallows the failure:
        with no GL context it returns normally and no window appears, which is
        exactly the 'it just doesn't come up' symptom. create_window() returns
        False, so the failure is reportable."""
        rospy.loginfo("Viewer: blue=reference orange=target(published T). "
                      "Close window to exit.")
        vis = o3d.visualization.Visualizer()
        try:
            ok = vis.create_window(
                window_name="LIVE result: blue=ref orange=target",
                width=1280, height=800)
        except Exception as e:
            rospy.logerr("create_window raised: %s: %s", type(e).__name__, e)
            ok = False
        if not ok:
            rospy.logerr("could not create a GL window -- this is a DISPLAY "
                         "problem, not a data problem.")
            rospy.logerr("  DISPLAY=%r  WAYLAND_DISPLAY=%r  SSH_CONNECTION=%r",
                         os.environ.get("DISPLAY"),
                         os.environ.get("WAYLAND_DISPLAY"),
                         os.environ.get("SSH_CONNECTION"))
            rospy.logerr("  test it standalone:  python3 -c \"import open3d as "
                         "o3d; o3d.visualization.draw_geometries("
                         "[o3d.geometry.TriangleMesh.create_sphere()])\"")
            rospy.logerr("  over SSH: reconnect with -X, or run on the console.")
            rospy.logerr("  on WSL:   export LIBGL_ALWAYS_SOFTWARE=1")
            try:
                vis.destroy_window()
            except Exception:
                pass
            return False
        for g in geoms:
            vis.add_geometry(g)
        vis.run()
        vis.destroy_window()
        return True

    def _fallback(self, ref_s, tgt_s):
        """No window: put the same overlay on disk so the result is still
        inspectable (CloudCompare, or another machine)."""
        merged = o3d.geometry.PointCloud(ref_s) + tgt_s
        path = os.path.join(self.out_dir, "overlay.pcd")
        o3d.io.write_point_cloud(path, merged)
        rospy.loginfo("wrote %s (blue=reference, orange=target) -- open it in "
                      "CloudCompare", path)
        np.savetxt(os.path.join(self.out_dir, "T_published.txt"), self.T,
                   fmt="%.9f")
        rospy.loginfo("wrote %s", os.path.join(self.out_dir, "T_published.txt"))
        try:
            png = os.path.join(self.out_dir, "overlay.png")
            r = o3d.visualization.rendering.OffscreenRenderer(1600, 1000)
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 2.0
            r.scene.add_geometry("ref", ref_s, mat)
            r.scene.add_geometry("tgt", tgt_s, mat)
            bounds = merged.get_axis_aligned_bounding_box()
            r.setup_camera(60.0, bounds, bounds.get_center())
            o3d.io.write_image(png, r.render_to_image())
            rospy.loginfo("wrote %s", png)
        except Exception as e:
            rospy.logwarn("offscreen render unavailable (%s: %s) -- the .pcd "
                          "is still there", type(e).__name__, e)


if __name__ == "__main__":
    try:
        LiveResultViz().run()
    except rospy.ROSInterruptException:
        pass