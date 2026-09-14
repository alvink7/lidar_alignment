# verify_pipeline findings

## Summary

- total queries: 64
- accepted: 0 (0%)
- no_match: 64

## Config

- ref_dir: /home/alvink/catkin_ws/maps/outdoor_keyframes
- bag: /home/alvink/catkin_ws/data/Apr4_Palio2.bag
- cloud_topic: /cloud_registered_body
- odom_topic: /Odometry
- query_every: 5
- tgt_accumulate_sec: 1.5
- yaw_sign: 1.0
- yaw_tol_deg: 30.0
- min_coverage: 0.25
- max_rmse: 0.3
- min_dyaw_deg: 0.5

## Per-query results

| q | index | sc_yaw | teaser_yaw | dyaw | coverage | rmse | result | nearest_idx | nearest_m | best_idx | sc_dist | q_n | q_extent | q_window_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 656.36 | -1 | - | - | - | - | - | no_match | 1 | 0.11 | 291 | 0.4624 | 3738 | 115.11 | 0.20 |
| 656.86 | -1 | - | - | - | - | - | no_match | 1 | 0.11 | 291 | 0.4913 | 4336 | 99.12 | 0.30 |
| 657.36 | -1 | - | - | - | - | - | no_match | 1 | 0.10 | 291 | 0.4652 | 4398 | 115.38 | 0.30 |
| 657.86 | -1 | - | - | - | - | - | no_match | 1 | 0.10 | 291 | 0.4644 | 4372 | 113.18 | 0.30 |
| 658.36 | -1 | - | - | - | - | - | no_match | 1 | 0.10 | 291 | 0.4649 | 4345 | 107.75 | 0.30 |
| 658.86 | -1 | - | - | - | - | - | no_match | 1 | 0.11 | 1 | 0.4624 | 4304 | 97.55 | 0.30 |
| 659.36 | -1 | - | - | - | - | - | no_match | 1 | 0.11 | 291 | 0.4728 | 3784 | 93.78 | 0.20 |
| 659.86 | -1 | - | - | - | - | - | no_match | 1 | 0.11 | 1 | 0.4556 | 3694 | 95.40 | 0.20 |
| 660.36 | -1 | - | - | - | - | - | no_match | 1 | 0.17 | 288 | 0.4794 | 4495 | 123.15 | 0.30 |
| 660.86 | -1 | - | - | - | - | - | no_match | 1 | 0.20 | 3 | 0.3905 | 3743 | 117.73 | 0.20 |
| 661.36 | -1 | - | - | - | - | - | no_match | 1 | 0.23 | 291 | 0.4716 | 4707 | 114.45 | 0.30 |
| 661.86 | -1 | - | - | - | - | - | no_match | 1 | 0.25 | 3 | 0.3514 | 3972 | 113.83 | 0.20 |
| 662.36 | -1 | - | - | - | - | - | no_match | 1 | 0.24 | 7 | 0.4224 | 3994 | 99.53 | 0.20 |
| 662.86 | -1 | - | - | - | - | - | no_match | 1 | 0.27 | 6 | 0.3581 | 4084 | 105.46 | 0.20 |
| 663.35 | -1 | - | - | - | - | - | no_match | 1 | 0.29 | 6 | 0.4010 | 4807 | 109.03 | 0.30 |
| 663.86 | -1 | - | - | - | - | - | no_match | 1 | 0.32 | 3 | 0.4281 | 4182 | 95.56 | 0.20 |
| 664.35 | -1 | - | - | - | - | - | no_match | 1 | 0.58 | 6 | 0.3490 | 4634 | 96.87 | 0.30 |
| 664.86 | -1 | - | - | - | - | - | no_match | 1 | 0.89 | 3 | 0.4038 | 4317 | 108.97 | 0.20 |
| 665.36 | -1 | - | - | - | - | - | no_match | 1 | 1.30 | 3 | 0.3714 | 3894 | 109.88 | 0.20 |
| 665.86 | -1 | - | - | - | - | - | no_match | 1 | 1.70 | 3 | 0.4502 | 4861 | 129.14 | 0.30 |
| 666.35 | -1 | - | - | - | - | - | no_match | 1 | 2.21 | 6 | 0.4824 | 4785 | 139.88 | 0.30 |
| 666.86 | -1 | - | - | - | - | - | no_match | 1 | 2.67 | 3 | 0.3814 | 4740 | 106.19 | 0.30 |
| 667.36 | -1 | - | - | - | - | - | no_match | 12 | 3.14 | 7 | 0.4266 | 3906 | 109.17 | 0.20 |
| 667.86 | -1 | - | - | - | - | - | no_match | 17 | 3.55 | 7 | 0.4224 | 4727 | 120.77 | 0.30 |
| 668.36 | -1 | - | - | - | - | - | no_match | 18 | 3.94 | 7 | 0.3728 | 4407 | 120.36 | 0.30 |
| 668.86 | -1 | - | - | - | - | - | no_match | 18 | 4.36 | 3 | 0.3726 | 4435 | 134.57 | 0.30 |
| 669.36 | -1 | - | - | - | - | - | no_match | 20 | 4.85 | 6 | 0.4200 | 3810 | 112.48 | 0.20 |
| 669.86 | -1 | - | - | - | - | - | no_match | 20 | 5.28 | 9 | 0.3996 | 4453 | 139.72 | 0.30 |
| 670.36 | -1 | - | - | - | - | - | no_match | 20 | 5.67 | 7 | 0.4196 | 3820 | 113.94 | 0.20 |
| 670.86 | -1 | - | - | - | - | - | no_match | 20 | 6.02 | 291 | 0.4997 | 4395 | 113.42 | 0.30 |
| 671.36 | -1 | - | - | - | - | - | no_match | 21 | 6.32 | 3 | 0.3897 | 4416 | 141.87 | 0.30 |
| 671.85 | -1 | - | - | - | - | - | no_match | 21 | 6.72 | 8 | 0.3799 | 4305 | 140.70 | 0.30 |
| 672.35 | -1 | - | - | - | - | - | no_match | 21 | 7.19 | 8 | 0.4164 | 4215 | 114.84 | 0.29 |
| 672.86 | -1 | - | - | - | - | - | no_match | 22 | 7.64 | 8 | 0.4698 | 3643 | 109.20 | 0.20 |
| 673.36 | -1 | - | - | - | - | - | no_match | 23 | 8.00 | 8 | 0.4160 | 4212 | 133.98 | 0.30 |
| 673.86 | -1 | - | - | - | - | - | no_match | 23 | 8.37 | 4 | 0.4387 | 3612 | 136.92 | 0.20 |
| 674.35 | -1 | - | - | - | - | - | no_match | 23 | 8.82 | 4 | 0.4079 | 4219 | 142.82 | 0.30 |
| 674.86 | -1 | - | - | - | - | - | no_match | 23 | 9.21 | 3 | 0.4505 | 4095 | 145.56 | 0.30 |
| 675.36 | -1 | - | - | - | - | - | no_match | 23 | 9.66 | 3 | 0.4372 | 3539 | 137.17 | 0.20 |
| 675.86 | -1 | - | - | - | - | - | no_match | 25 | 10.05 | 4 | 0.4497 | 4034 | 137.39 | 0.30 |
| 676.36 | -1 | - | - | - | - | - | no_match | 25 | 10.46 | 3 | 0.4578 | 3423 | 136.49 | 0.20 |
| 676.86 | -1 | - | - | - | - | - | no_match | 27 | 10.81 | 3 | 0.5037 | 4091 | 137.66 | 0.30 |
| 677.36 | -1 | - | - | - | - | - | no_match | 27 | 11.13 | 4 | 0.4601 | 3519 | 135.96 | 0.20 |
| 677.86 | -1 | - | - | - | - | - | no_match | 27 | 11.46 | 4 | 0.4404 | 4189 | 136.27 | 0.30 |
| 678.36 | -1 | - | - | - | - | - | no_match | 31 | 11.76 | 4 | 0.4844 | 3591 | 135.54 | 0.20 |
| 678.86 | -1 | - | - | - | - | - | no_match | 31 | 12.06 | 3 | 0.4660 | 3690 | 122.18 | 0.20 |
| 679.36 | -1 | - | - | - | - | - | no_match | 31 | 12.32 | 3 | 0.4058 | 3704 | 133.84 | 0.20 |
| 679.86 | -1 | - | - | - | - | - | no_match | 31 | 12.65 | 4 | 0.4021 | 4338 | 134.63 | 0.30 |
| 680.36 | -1 | - | - | - | - | - | no_match | 31 | 13.04 | 3 | 0.3977 | 4452 | 132.62 | 0.30 |
| 680.86 | -1 | - | - | - | - | - | no_match | 31 | 13.42 | 3 | 0.3588 | 4422 | 130.44 | 0.30 |
| 681.36 | -1 | - | - | - | - | - | no_match | 31 | 13.85 | 4 | 0.3595 | 3765 | 135.80 | 0.20 |
| 681.86 | -1 | - | - | - | - | - | no_match | 31 | 14.33 | 3 | 0.4089 | 4421 | 124.80 | 0.30 |
| 682.36 | -1 | - | - | - | - | - | no_match | 31 | 14.78 | 3 | 0.4040 | 4434 | 130.44 | 0.30 |
| 682.86 | -1 | - | - | - | - | - | no_match | 31 | 15.17 | 3 | 0.4118 | 3892 | 121.10 | 0.20 |
| 683.36 | -1 | - | - | - | - | - | no_match | 31 | 15.47 | 3 | 0.4142 | 4512 | 119.29 | 0.30 |
| 683.86 | -1 | - | - | - | - | - | no_match | 31 | 15.67 | 3 | 0.4361 | 3979 | 130.85 | 0.20 |
| 684.36 | -1 | - | - | - | - | - | no_match | 31 | 15.81 | 3 | 0.4199 | 4692 | 129.23 | 0.30 |
| 684.86 | -1 | - | - | - | - | - | no_match | 31 | 15.79 | 3 | 0.4250 | 4054 | 134.86 | 0.20 |
| 685.36 | -1 | - | - | - | - | - | no_match | 31 | 15.86 | 3 | 0.4160 | 4450 | 127.96 | 0.30 |
| 685.86 | -1 | - | - | - | - | - | no_match | 31 | 15.99 | 3 | 0.4583 | 4792 | 130.14 | 0.30 |
| 686.35 | -1 | - | - | - | - | - | no_match | 31 | 16.05 | 3 | 0.4184 | 4636 | 115.27 | 0.30 |
| 686.86 | -1 | - | - | - | - | - | no_match | 31 | 16.06 | 3 | 0.4416 | 4754 | 136.72 | 0.30 |
| 687.36 | -1 | - | - | - | - | - | no_match | 31 | 16.13 | 2 | 0.3552 | 3730 | 113.12 | 0.20 |
| 687.86 | -1 | - | - | - | - | - | no_match | 31 | 16.28 | 3 | 0.4823 | 4766 | 120.03 | 0.30 |

## Interpretation notes

- **yaw_disagreement**, clustered around one consistent sign flip -> wrong
  `--yaw_sign`; re-run `scancontext/tests/test_yaw_convention.py` and match
  its printed convention.
- **quality_gate** -> retrieval is likely landing on the right keyframe but
  GICP refinement is weak; tune `ceiling_crop_m` / `pc_max_radius` / keyframe
  spacing in the reference DB build (`sc_reference.py`).
- **no_match** -> the query is out-of-map, the reference DB is too sparse,
  or the query recipe (`accumulate_sec`/`voxel`/`ceiling_crop_m`) does not
  match the recipe the DB was built with (`db.sc_params`).
- **align_error** -> align_target raised (e.g. too few FPFH correspondences
  on a sparse/degenerate target window); same failure mode
  `sc_relocalize._main()`'s `--relocalize` guard already anticipates.

Diagnostic columns (`nearest_idx`/`nearest_m`/`best_idx`/`sc_dist`/`q_n`/`q_extent`)
explain WHY, independent of the gates above:
- `no_match` with small `nearest_m` (a keyframe really was right there) but
  large `sc_dist` -> the descriptors genuinely don't agree at the same
  place -> the query recipe likely differs from the DB build recipe
  (compare `db.sc_params` to what `iterate_queries`/`build_sc_query_cloud`
  apply -- accumulate_sec/voxel/ceiling_crop_m must match exactly).
- `no_match` with `sc_dist` just above the DB's `sc_dist_thres` -> the
  threshold is too strict for this scene; loosen `--sc_dist_thres` when
  rebuilding the DB (`sc_reference.py`).
- `no_match` with large `nearest_m` -> genuinely out-of-map; expected.
- `best_idx` != `nearest_idx` on a low-`sc_dist` row -> retrieval landed on
  a descriptor-similar but spatially-wrong keyframe (a symmetry/aliasing
  trap, not a recipe bug).
- `q_window_s` noticeably larger than `accumulate_sec` (config above) ->
  the window is NOT being trimmed correctly -- it's spanning real travel
  (a trajectory smear, not one viewpoint) -- a guaranteed no_match
  regardless of anything else. `q_extent` (raw XY span) is a weaker signal
  for this on a long-range outdoor sensor, where even a single viewpoint
  can show tens-to-hundreds of metres of extent from real sensor range;
  trust `q_window_s` over `q_extent` here. Run with `--self_check` first: a
  clean diagonal (every keyframe retrieves itself at ~0 distance) rules out
  the retrieval core (tree/exclusion/keys) entirely and confirms the
  problem is recipe/window drift between DB build and query, not the
  matcher itself.
