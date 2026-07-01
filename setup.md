# cloud_aligner — Environment Setup

Do this once per machine that will run the node. The dependencies are GPU- and
Python-version-specific compiled code, so each machine builds/installs its own —
nothing binary is committed to the repo.

All of this must target the **same Python3 that ROS Noetic uses** (usually
`/usr/bin/python3`, Python 3.8 on Ubuntu 20.04). Verify:

```bash
which python3            # expect /usr/bin/python3
python3 --version        # note your version — used in the TEASER paths below
```

> Commands below assume Python 3.8 (Noetic default). If `python3 --version`
> shows something else, adjust the version numbers in the TEASER++ install
> paths (section 2c) accordingly — a version-agnostic form is given there.

---

## 1. pip dependencies

```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

This installs `numpy`, `open3d`, `small-gicp`, and `faiss-gpu`.

- **No NVIDIA GPU / CUDA?** Edit `requirements.txt`: replace `faiss-gpu` with
  `faiss-cpu`. (The node's FAISS calls will then run on CPU; correctness is
  unchanged, speed is lower.)
- Confirm FAISS sees the GPU:
  ```bash
  python3 -c "import faiss; print('GPUs:', faiss.get_num_gpus())"
  ```

---

## 2. TEASER++ (build from source — no pip package)

TEASER++ has no pip wheel; the Python binding is compiled against this machine's
Python and Eigen. **Build it with your ROS Python active.**

### 2a. System + pybind11 prerequisites

```bash
sudo apt install cmake libeigen3-dev libboost-all-dev
python3 -m pip install --user "pybind11[global]"
python3 -c "import pybind11; print(pybind11.get_cmake_dir())"   # must print a path
```

> The apt `pybind11-dev` is too old for current TEASER++ master — use the pip
> `pybind11[global]` above, which ships the CMake config files.

### 2b. Clone and build the Python binding

```bash
git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus
mkdir build && cd build

cmake -DBUILD_PYTHON_BINDINGS=ON \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") \
      ..

make teaserpp_python -j$(nproc)
```

Confirm in the cmake output you saw the binding get enabled (a line like
"TEASER++ Python binding will be built"). The make step must show real
compilation (`Building CXX object ...` / `Linking ...`), ending with
`_teaserpp.cpython-38-...-linux-gnu.so` in `build/python/teaserpp_python/`.

### 2c. Install the binding into your user site-packages

This version of TEASER++ produces only the compiled `.so` (no `setup.py`), so
assemble the package manually: the `.so` plus the source-tree `__init__.py`
must sit together in a `teaserpp_python/` folder on the Python path.

```bash
SITE=~/.local/lib/python3.8/site-packages/teaserpp_python
mkdir -p "$SITE"
cp build/python/teaserpp_python/_teaserpp.cpython-38-*-linux-gnu.so "$SITE"/
cp ../python/teaserpp_python/__init__.py "$SITE"/
```

> **Not on Python 3.8?** The paths above are Python-3.8-specific (Ubuntu 20.04 /
> ROS Noetic default). On another version, adjust both:
> - the site-packages path: `python3.8` → your version (e.g. `python3.10`), or
>   just use `$(python3 -c "import site; print(site.getusersitepackages())")`
>   to get it automatically;
> - the `.so` filename: `cpython-38` → your version (e.g. `cpython-310`). The
>   build produces whatever matches the Python you compiled against — check the
>   actual name with `ls build/python/teaserpp_python/*.so`.
>
> Version-agnostic form:
> ```bash
> SITE=$(python3 -c "import site; print(site.getusersitepackages())")/teaserpp_python
> mkdir -p "$SITE"
> cp build/python/teaserpp_python/_teaserpp*.so "$SITE"/
> cp ../python/teaserpp_python/__init__.py "$SITE"/
> ```

### 2d. Verify (from a directory OTHER than the build tree)

```bash
cd ~
python3 -c "import teaserpp_python; print('TEASER OK'); print(teaserpp_python.RobustRegistrationSolver)"
```

`TEASER OK` plus a class reference means it's installed.

---

## 3. Final import check

All five must import under the ROS Python before the node will run:

```bash
/usr/bin/python3 -c "import rospy, faiss, teaserpp_python, small_gicp, open3d; print('all imports OK')"
```

(`rospy` requires `source /opt/ros/noetic/setup.bash` first.)

---

## 4. Build the package

```bash
cd ~/catkin_ws
catkin_make            # or: catkin_make --pkg cloud_aligner
source devel/setup.bash
rospack find cloud_aligner
```

You're ready — see **USAGE.md**.

---

## Notes / troubleshooting

- **`import teaserpp_python` segfaults** → the binding was built against a
  different Python than the one importing it. Rebuild with
  `-DPYTHON_EXECUTABLE=/usr/bin/python3` and reinstall (2b–2c).
- **`No module named 'pybind11'` during cmake** → run the pip `pybind11[global]`
  install (2a), then `rm -rf build` and reconfigure fresh (stale CMake cache
  ignores the fix otherwise).
- **`faiss.swigfaiss_avx2` warning at import** → harmless; FAISS falls back to
  the non-AVX2 build and works fine.
- **`No module named 'alignment_core'` when running the node** → not an env
  issue; the node already adds its own dir to the path. Make sure
  `alignment_core.py` and `ros_cloud_utils.py` are in `scripts/` next to
  `alignment_node.py`.