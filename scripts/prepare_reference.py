#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_reference.py
--------------------
One-time offline step: read the REFERENCE bag, accumulate its /cloud_registered
frames into a single dense cloud, and save it as a PCD that alignment_node.py
loads at startup.

This keeps the reference "preloaded" — the live node never re-derives it from a
bag. The node applies levelling/FPFH preprocessing itself (so the saved PCD is
just the raw accumulated reference cloud in its native frame).

Usage:
  python prepare_reference.py \
      --bag ~/catkin_ws/data/Apr4_droneparkscan.bag \
      --topic /cloud_registered \
      --every_nth 15 \
      --out ~/maps/reference_prepared.pcd

NOTE: "larger bag = reference" per current instruction. Point --bag at whichever
file is the reference.
"""

import argparse
import os
import numpy as np
import open3d as o3d

import rosbag
from sensor_msgs import point_cloud2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--every_nth", type=int, default=10)
    ap.add_argument("--out", default=os.path.expanduser("../maps/test_tgt1.pcd"))
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="optional voxel downsample of saved reference (0=off)")
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
