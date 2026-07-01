# cloud_aligner

ROS1 (Noetic) package that aligns a live point-cloud stream to a preloaded
reference cloud and publishes the rigid transform between them.

The published transform maps points from the **target** frame into the
**reference** frame.

## Pipeline

accumulate first N `/cloud_registered` frames → ICP-voxel downsample → ground
level → ground-remove → FPFH → FAISS-GPU mutual-NN → TEASER++ global
registration → small_gicp GICP refinement → publish 4×4 (once).

## Layout

```
cloud_aligner/
  scripts/
    alignment_core.py        # ROS-free pipeline (shared, unit-testable)
    ros_cloud_utils.py        # PointCloud2 <-> open3d, matrix -> TransformStamped
    alignment_node.py         # live node: accumulate N, align, publish, done
    prepare_reference.py      # one-time: bag -> reference PCD
    visualize_alignment.py    # offline overlay check (recomputes transform)
    visualize_live_result.py  # overlay using the node's PUBLISHED matrix
    diagnose_alignment.py     # ground normals + multi-threshold overlap report
    alignment_quality.py      # (in progress) geometric quality gate
  config/params.yaml          # all tunable parameters
  launch/align.launch
  package.xml
  CMakeLists.txt
```

## Dependencies

`open3d`, `numpy`, `faiss` (GPU), `teaserpp_python` (built from source),
`small_gicp`, plus ROS `rospy`, `sensor_msgs`, `geometry_msgs`, `tf`.
All must be importable by the Python3 that ROS uses (`/usr/bin/python3`).

## Build

```bash
cd ~/catkin_ws
catkin_make            # or: catkin_make --pkg cloud_aligner
source devel/setup.bash
rospack find cloud_aligner    # should print the path
```

## Configure

Edit `config/params.yaml` — no need to touch Python. The most-tuned values
(`reference_pcd`, `input_topic`, `num_frames`, `reference_frame`) can also be
overridden on the command line, which takes precedence over the yaml.

## Use

### 1. Prepare the reference (once)

"Larger / more complete scan = reference."

```bash
rosrun cloud_aligner prepare_reference.py \
    --bag ~/catkin_ws/data/<reference>.bag \
    --topic /cloud_registered --every_nth 15 \
    --out ~/catkin_ws/maps/reference1.pcd
```

### 2. Run

Bag playback requires sim time so TF timestamps line up:

```bash
# Terminal A
roscore
rosparam set use_sim_time true            # BEFORE starting nodes

# Terminal B
roslaunch cloud_aligner align.launch

# Terminal C
rosbag play --clock ~/catkin_ws/data/<target>.bag
```

Override any arg inline, e.g. `roslaunch cloud_aligner align.launch num_frames:=30`.

### 3. Outputs

- **Static TF** `reference_map -> camera_init` (on `/tf_static`) — the main
  output; composes with FAST-LIO's `camera_init -> body` so the drone's pose is
  expressed in the reference frame automatically.
- `/target_to_reference` — `geometry_msgs/TransformStamped` (latched)
- `/target_to_reference_matrix` — `std_msgs/Float64MultiArray`, 4x4 row-major (latched)

Verify the chain (with bag playing, node fired):

```bash
rosrun tf tf_echo reference_map camera_init   # constant static link
rosrun tf tf_echo reference_map body          # updates as drone moves
```

### 4. Check it visually

Offline (recomputes from PCDs):

```bash
rosrun cloud_aligner visualize_alignment.py REF.pcd TGT.pcd
# (pure open3d; can also run directly: python3 scripts/visualize_alignment.py ...)
```

Live (uses the node's actual published matrix):

```bash
rosrun cloud_aligner visualize_live_result.py \
    _reference_pcd:=$HOME/catkin_ws/maps/reference1.pcd \
    _target_pcd:=$HOME/catkin_ws/maps/<target>.pcd
```

Blue = reference, orange = transformed target. Overlap = good.

## Notes

- **Use `/cloud_registered`, not `/livox/lidar`.** The pipeline accumulates
  frames, which only works on registered (FAST-LIO2 output) clouds that already
  share a global frame.
- **Bag runs need `use_sim_time true` + `rosbag play --clock`**, or TF/RViz will
  report frames as "not part of the same tree" even when the transform is correct.
- **`body` far from the `reference_map` origin is normal** — that's the drone's
  real position in the map. Judge alignment by whether the live scan *overlaps*
  the reference cloud, not by frame position.
- **Fitness scalars are unreliable here.** TEASER/GICP fitness read very low
  (<1%) even on good outdoor alignments. Judge by the visual overlay, not the
  fitness number.
- **`num_frames` is the main tuning knob.** If a live alignment looks off,
  raise it — the first frames can be sparse.
- **One-shot, no drift tracking.** The static TF is fixed once from the first N
  frames; it does not correct FAST-LIO drift during a long run.
- Reference preprocessing runs once at startup; expect a few seconds before
  "Listening".

See **USAGE.md** for full run/verify/visualize steps and **SETUP.md** for
dependency install.