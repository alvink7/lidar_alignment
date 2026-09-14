#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
the reference should generally be the larger bag so we don't spend more online time doing computations

--sc_out_dir (optional): ALSO build a Scan Context keyframe DB from the
same bag, via scancontext/python/sc_reference.build_reference_db(). This
is a SEPARATE index from the fused .pcd below -- Scan Context compares an
egocentric query against a database of egocentric observations, and a
fused multi-frame cloud has no single vantage point to compare against
(see scancontext/python/sc_reference.py docstring). The fused .pcd this
script has always produced is unaffected and still useful for dense GICP
geometry; it's just not what Scan Context indexes.
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d

import rosbag
from sensor_msgs import point_cloud2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "..", "scancontext", "python"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--every_nth", type=int, default=10)
    ap.add_argument("--out", default=os.path.expanduser("/home/alvink/catkin_ws/maps/reference_office.pcd"))
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="optional voxel downsample of saved reference (0=off)")

    # -------- optional Scan Context keyframe DB (see module docstring) --------
    ap.add_argument("--sc_out_dir", default=None,
                     help="if set, ALSO build a Scan Context keyframe DB "
                          "here from this bag's /cloud_registered_body + "
                          "/Odometry")
    ap.add_argument("--sc_cloud_topic", default="/cloud_registered_body")
    ap.add_argument("--sc_odom_topic", default="/Odometry")
    ap.add_argument("--sc_lidar_height", type=float, default=None,
                     help="required if --sc_out_dir is set: the sensor's "
                          "approximate height above ground for THIS "
                          "platform/mounting (metres) -- see "
                          "sc_keyframes.build_sc_cloud docstring")
    ap.add_argument("--sc_spacing_m", type=float, default=0.5)
    ap.add_argument("--sc_spacing_deg", type=float, default=10.0)
    ap.add_argument("--sc_accumulate_sec", type=float, default=0.3)
    ap.add_argument("--sc_voxel", type=float, default=0.5)
    ap.add_argument("--sc_ceiling_crop_m", type=float, default=None,
                     help="metres above the SENSOR; set indoors/under cover")
    ap.add_argument("--sc_pc_max_radius", type=float, default=20.0,
                     help="size to the actual environment scale")
    ap.add_argument("--sc_dist_thres", type=float, default=0.13)
    args = ap.parse_args()

    if args.sc_out_dir and args.sc_lidar_height is None:
        raise SystemExit("--sc_lidar_height is required when --sc_out_dir is set "
                          "(no default -- it's a real, platform-specific value)")

    bag_path = os.path.expanduser(args.bag)
    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.isfile(bag_path):
        raise SystemExit("Bag not found: %s" % bag_path)

    print("Reading %s  topic=%s  every_nth=%d" %
          (bag_path, args.topic, args.every_nth))

    bag = rosbag.Bag(bag_path, "r")
    total = bag.get_message_count(topic_filters=[args.topic])
    print("Messages on topic: %d" % total)

    all_pts = []
    kept = 0
    for idx, (topic, msg, t) in enumerate(
            bag.read_messages(topics=[args.topic])):
        if idx % args.every_nth != 0:
            continue
        pts = np.array(
            list(point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float64)
        if pts.size:
            all_pts.append(pts[:, :3])
            kept += 1
        if kept % 20 == 0 and kept > 0:
            print("  kept %d frames..." % kept)
    bag.close()

    if not all_pts:
        raise SystemExit("No points extracted — check topic name.")

    merged = np.vstack(all_pts)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(merged)
    print("Accumulated %d frames -> %d points" % (kept, len(cloud.points)))

    if args.voxel > 0:
        before = len(cloud.points)
        cloud = cloud.voxel_down_sample(args.voxel)
        print("Voxel %.3f m: %d -> %d points" %
              (args.voxel, before, len(cloud.points)))

    o3d.io.write_point_cloud(out_path, cloud)
    print("Saved reference: %s" % out_path)

    if args.sc_out_dir:
        import sc_reference
        print("\nBuilding Scan Context keyframe DB -> %s ..." % args.sc_out_dir)
        n_kf = sc_reference.build_reference_db(
            bag_path, os.path.expanduser(args.sc_out_dir), args.sc_lidar_height,
            cloud_topic=args.sc_cloud_topic, odom_topic=args.sc_odom_topic,
            spacing_m=args.sc_spacing_m, spacing_deg=args.sc_spacing_deg,
            accumulate_sec=args.sc_accumulate_sec, voxel=args.sc_voxel,
            ceiling_crop_m=args.sc_ceiling_crop_m,
            pc_max_radius=args.sc_pc_max_radius, sc_dist_thres=args.sc_dist_thres)
        print("Saved %d Scan Context keyframes -> %s" % (n_kf, args.sc_out_dir))


if __name__ == "__main__":
    main()
