#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from sensor_msgs import point_cloud2


def pointcloud2_to_o3d(msg):
    """Extract XYZ from a PointCloud2 into an open3d cloud."""
    pts = np.array(
        list(point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float64,
    )
    cloud = o3d.geometry.PointCloud()
    if pts.size:
        cloud.points = o3d.utility.Vector3dVector(pts[:, :3])
    return cloud


def accumulate_clouds(clouds):
    """Concatenate a list of o3d clouds into one."""
    if not clouds:
        return o3d.geometry.PointCloud()
    all_pts = np.vstack([np.asarray(c.points) for c in clouds
                         if len(c.points) > 0])
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(all_pts)
    return out


def matrix_to_transform_stamped(T, parent_frame, child_frame, stamp):
    """4x4 numpy -> geometry_msgs/TransformStamped.

    The transform expresses child_frame's pose in parent_frame, i.e. it maps
    points from child_frame into parent_frame. Here parent=reference,
    child=target.
    """
    from geometry_msgs.msg import TransformStamped
    import tf.transformations as tft

    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent_frame      # reference
    ts.child_frame_id = child_frame        # target

    ts.transform.translation.x = float(T[0, 3])
    ts.transform.translation.y = float(T[1, 3])
    ts.transform.translation.z = float(T[2, 3])

    q = tft.quaternion_from_matrix(T)      # (x, y, z, w)
    ts.transform.rotation.x = float(q[0])
    ts.transform.rotation.y = float(q[1])
    ts.transform.rotation.z = float(q[2])
    ts.transform.rotation.w = float(q[3])
    return ts
