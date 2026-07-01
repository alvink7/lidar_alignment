# cloud_aligner — Usage Guide

** disclaimer: I had Claude generate this usage documentation, and it starts talking about rviz tests for visual verification, which I could not get working (I suspect rviz isn't grabbing the tf between the reference and body frame properly, but this is worth looking into later), so please ignore that part for now. There is also no quality gate on this computation, meaning we use the first computed transform by default. The reason for this was that the fitness calculations on the alignment is artifically botched because of the randomness from outdoor vegetation (in other words, there is no way I can think of to reliably reject a transform without visually verifying it). For early stage tests in indoor environments, I imagine it is still a decent idea to implement this filter. 

ROS1 (Noetic) package that aligns a live point-cloud stream to a preloaded
reference cloud, publishes the rigid transform between them, and broadcasts it
as a static TF so FAST-LIO's odometry is expressed in the reference frame.

The published transform maps points from the **target** frame into the
**reference** frame.

---

## File descriptions

- **`alignment_node.py`** — the ROS node. Subscribes to `/cloud_registered`,
  accumulates the first *N* frames (default 10), runs alignment once, publishes
  the transform, and broadcasts a static TF `reference_frame -> camera_init`.
  Configuration is read from `config/params.yaml`.
- **`alignment_core.py`** — all alignment logic and algorithms. Everything other
  than point-cloud accumulation lives here. Requires `faiss` (faiss-gpu),
  `teaserpp_python`, and `small_gicp`.
- **`ros_cloud_utils.py`** — helpers to convert rosbag/ROS messages into formats
  the algorithms use (PointCloud2 -> open3d, 4x4 -> TransformStamped).
- **`prepare_reference.py`** — one-time offline step: reference bag -> reference PCD.

---

## One-time setup

See **SETUP.md** for full dependency install (faiss, small_gicp, and the
from-source TEASER++ build). Then build:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rospack find cloud_aligner
```

For RViz reference visualization also install pcl_ros:

```bash
sudo apt install ros-noetic-pcl-ros
```

> Rebuild only when CMakeLists/package.xml change or you add a node. Editing
> Python or params.yaml needs no rebuild. Every new terminal needs
> `source devel/setup.bash`.

---

## How to use

### Step 1 — Prepare the reference (once)

Pick the larger / more complete scan as the reference.

```bash
rosrun cloud_aligner prepare_reference.py \
    --bag ~/catkin_ws/data/<reference>.bag \
    --topic /cloud_registered --every_nth 15 \
    --out ~/catkin_ws/maps/reference1.pcd
```

Make sure `reference_pcd` in params.yaml matches `--out`.

### Step 2 — Run (bags need sim time)

**Because you are replaying a bag, the whole system must share the bag's clock.**
Set sim time before starting nodes, and play the bag with `--clock`:

```bash
# Terminal A
roscore
rosparam set use_sim_time true       # BEFORE starting nodes

# Terminal B — the node
roslaunch cloud_aligner align.launch

# Terminal C — play the target bag WITH --clock
rosbag play --clock ~/catkin_ws/data/<target>.bag
```

Without `use_sim_time` + `--clock`, TF timestamps won't match and RViz will
report the reference and camera_init frames as "not part of the same tree"
even though the transform is correct.

Override any tuned value inline, e.g. `roslaunch cloud_aligner align.launch num_frames:=30`.

### Step 3 — Outputs

The node produces three things once alignment fires:

- **Static TF** `reference_map -> camera_init` (on `/tf_static`) — this is the
  main output. TF composes it with FAST-LIO's `camera_init -> body` so the
  drone's live pose is expressed in the reference frame automatically.
- `/target_to_reference` — `geometry_msgs/TransformStamped` (latched).
- `/target_to_reference_matrix` — `std_msgs/Float64MultiArray`, 4x4 row-major (latched).

```bash
rostopic echo /tf_static                    # the static transform
rostopic echo /target_to_reference_matrix   # the raw 4x4
```

### Step 4 — Verify the TF chain (definitive functional test)

With the bag playing (and the node having fired):

```bash
# static link — should be CONSTANT
rosrun tf tf_echo reference_map camera_init

# composed pose — should UPDATE as the drone moves
rosrun tf tf_echo reference_map body
```

`reference_map -> body` printing a smoothly-updating pose that stays within the
reference map's extent means the whole pipeline works: alignment, static TF, and
FAST-LIO composition.

> Before alignment fires, `reference_map -> body` will fail with "cannot find
> transform" — that's expected; it resolves once the node broadcasts.

### Step 5 — Visualize in RViz

```bash
rviz
```

1. **Global Options -> Fixed Frame -> `reference_map`.**
2. **Add -> By topic -> `/cloud_registered` -> PointCloud2** (the live scan).
3. Publish the reference cloud into the reference frame and add it:
   ```bash
   rosrun pcl_ros pcd_to_pointcloud ~/catkin_ws/maps/reference1.pcd 0.5 \
       _frame_id:=reference_map
   ```
   Then **Add -> By topic -> `/cloud_pcd` -> PointCloud2**.
4. Give each cloud a Flat Color (e.g. reference=blue, live=orange).

**Success = the live scan overlaps the reference cloud** (same walls, ground,
structures). The `body` frame being offset from the `reference_map` origin is
normal — that's just where the drone physically is. Judge by cloud overlap, not
by the frame's distance from the origin.

If RViz still shows "not part of the same tree" after alignment fired: you almost
certainly forgot `use_sim_time`/`--clock` (Step 2), or RViz cached the
disconnected state — restart RViz after the node has broadcast.

---

## Configuration & tuning

All values live in **`config/params.yaml`** (no rebuild to change). The four
most-common (`reference_pcd`, `input_topic`, `num_frames`, `reference_frame`)
can also be overridden on the roslaunch line, which wins over the yaml.

### Main knob: `num_frames`

| Symptom | Action |
|---|---|
| Live scan offset from reference in RViz / sparse first frames | **Raise** `num_frames` (10 -> 30) |
| Want lowest latency, dense sensor | **Lower** toward 1 |

The live transform comes from only the first *N* frames, which can be thinner
than a whole-bag offline test. If the live overlay is worse than an offline
`visualize_alignment.py` run on the same data, raise `num_frames` first.

### Pipeline parameters

| Param | Default | Effect | Tune when |
|---|---|---|---|
| `RANSAC_VOXEL` | 0.30 | FPFH downsample. Smaller -> more correspondences, slower. | `<3 correspondences` error -> lower to 0.15-0.20 |
| `TEASER_NOISE_BOUND` | 0.60 | Max match error (m); ~ `RANSAC_VOXEL x 2`. | Change with `RANSAC_VOXEL` |
| `GROUND_REMOVE_Z` | 0.30 | Height below which points are dropped as ground. | Raise if clutter pollutes matching; lower if real structure removed |
| `ICP_VOXEL` | 0.10 | ICP refinement downsample. | Rarely |
| `ICP_DIST_COARSE` | 0.5 | ICP coarse correspondence dist (m). | Raise if global init rough |
| `ICP_DIST_FINE` | 0.1 | ICP fine correspondence dist (m). | Lower for tighter fit on clean data |

### TF / frame parameters

| Param | Default | Meaning |
|---|---|---|
| `broadcast_tf` | true | Broadcast the static TF. False = publish topics only. |
| `odom_parent_frame` | `camera_init` | FAST-LIO's odometry parent frame. **Must match** FAST-LIO's actual frame — confirm with `rostopic echo /Odometry -n1 \| grep frame_id`. |
| `reference_frame` | `reference_map` | Parent frame on the published/broadcast transform. |
| `target_frame` | (from msg header) | Child frame for the topic message; auto from incoming cloud. |

---

## Judging quality (important)

**Do not trust the TEASER/GICP fitness numbers.** On sparse outdoor LiDAR they
read very low (often <1%) even on good alignments, and GICP RMSE can print
`nan`/`inf`. This is expected, not failure.

Judge by the **RViz overlay** (Step 5) — clouds overlapping = good. For a
geometric report when something looks wrong:

```bash
python3 ~/catkin_ws/src/cloud_aligner/scripts/diagnose_alignment.py REF.pcd TGT.pcd
```

It prints the two ground-plane normals and an overlap-vs-distance curve. Good
alignment = high overlap already at a tight threshold (>=60-70% within 0.5 m).

---

## Limitations

- **One-shot, no drift tracking.** The static TF is computed once from the first
  N frames and stays fixed. It corrects FAST-LIO's initial placement but rides
  along with any drift FAST-LIO accumulates during a long run. Correcting
  ongoing drift needs continuous re-alignment (not implemented).
- Reference preprocessing runs once at startup; expect a few seconds before
  "Listening".

---

## Common pitfalls

- **Use `/cloud_registered`, not `/livox/lidar`.** The node accumulates frames,
  which only works on registered (FAST-LIO2 output) clouds sharing a global frame.
- **Bag runs need `use_sim_time true` + `rosbag play --clock`** or TF/RViz frames
  won't connect (see Step 2).
- **`odom_parent_frame` must match FAST-LIO** (`camera_init` by default).
- **RViz "not part of the same tree"** -> sim-time issue, or restart RViz after
  the node has broadcast the static TF.
- **`body` far from `reference_map` origin is normal** — that's the drone's real
  position. Judge alignment by cloud overlap, not frame position.
- **New terminal can't find the package** -> `source devel/setup.bash`.
- **`env: 'python'` error** -> scripts need `#!/usr/bin/env python3`; all shipped
  ones already do.