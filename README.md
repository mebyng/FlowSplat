# FlowSplat

FlowSplat is an experimental synthetic-data pipeline for camera-conditioned novel
view synthesis. The current implementation trains image-space RGB or RGB-D models
and back-projects depth into colored point clouds for interactive inspection.

The longer-term goal is **single image → novel views → 3D Gaussian scene**.
The viewer supports point clouds and trained Gaussian scenes; Gaussian training
uses a frozen flow teacher.

## Current implementation

- **Synthetic data:** procedural scenes containing boxes and spheres, rendered
  with Trimesh/Pyrender. Random camera views save RGB, float32 depth, camera-to-world
  poses, and intrinsics. Lighting is a fixed directional light.
- **View pairs:** source and target come from the same scene. Training samples a
  different target randomly; validation uses the next view deterministically.
- **`generate` mode:** conditional flow matching from Gaussian noise to the target
  image, conditioned on a source view and relative camera rotation. Sampling uses
  Euler integration (10 steps by default).
- **`rotate` mode:** deterministic regression from a source view and camera rotation
  to the target image.
- **Network:** convolutional encoder/decoder with three bottleneck cross-attention
  blocks and camera/time embeddings. Both view models use MSE; RGB-D treats raw
  depth as a fourth channel with the same loss weighting as RGB.
- **Autoencoder/latents:** a convolutional autoencoder and an offline encoding
  script exist, but the training and latent workflows need fixes (see TODOs).
- **Training:** Adam, validation, TensorBoard metrics and image/depth panels,
  gradient clipping, warmup, scheduling, run metadata, and resumable checkpoints.
- **Viewer:** OpenGL orbit viewer for ground-truth RGB-D point clouds. A trained raw
  RGB-D run can predict additional views and append their points to the cloud.

The flow objective uses `x_t = (1 - t) * noise + t * target` and velocity target
`target - noise`. It does not interpolate between the source and target images;
the source is a separate conditioning input.

## Setup

Python **3.11+** is required. Dependencies and the lockfile are in
[`pyproject.toml`](pyproject.toml) and `uv.lock`.

```bash
uv sync
source .venv/bin/activate
```

Training currently requires a CUDA-capable PyTorch installation because `main.py`
uses `.cuda()` directly. Rendering needs a working OpenGL context, and the
interactive viewer needs a graphical display. No measured VRAM budget or real-time
performance guarantee is established in this repository.

## Generate a dataset

```bash
python data/generate_dataset.py
```

With no options, the script writes `datasets/scenes_10000_10_colored_spheres_depth/` with:

- 10,000 training scenes and 10 validation scenes;
- 10 random views per scene at 128×128, with camera radius 2.0;
- three randomly positioned, fixed-radius colored spheres per scene.

Use CLI options to change the settings. For a smaller, seeded dataset:

```bash
python data/generate_dataset.py \
  --output-dir datasets/demo --training-scenes 100 --validation-scenes 10 \
  --num-views 10 --resolution 128 --radius 2.0 --num-objects 3 \
  --random-type --random-size --seed 42
```

Set either scene count to `0` to skip that split. Randomization switches support
both forms: `--random-size` / `--no-random-size`, `--random-color` /
`--no-random-color`, and `--random-type` / `--no-random-type`. Disabled random type
uses spheres only. The CLI always generates random camera views. Grid rendering
remains available through `create_dataset(random_views=False, step_z=..., step_x=...)`,
but currently omits the poses and depth required by training.
Run `python data/generate_dataset.py --help` for all options.

```text
datasets/<name>/
  training/scene_000/
    render_rand_0000.png
    depth_rand_0000.npz
    camera_matrices.csv
    intrinsics.csv
  validation/scene_000/
    ...
```

`camera_matrices.csv` records image/depth filenames and pose entries `m0`…`m15`.
`intrinsics.csv` holds a 3×3 matrix. Depth is camera-forward z-depth, with zero
background values. Optional precomputed latents use `.pt` files referenced by an
`encoded_filename` CSV column. Older RGB-only datasets must be regenerated to use
`--use-depth`.

## Train and inspect runs

From the repository root:

```bash
python main.py \
  --dataset-path datasets/scenes_10000_10_colored_spheres_depth \
  --mode generate --use-depth --epochs 50 --batch-size 16 \
  --lr 0.0003 --grad-clip 1.0 --warmup-epochs 2 \
  --seed 42 --validation-seed 0 --run-name rgbd_flow_seed42

tensorboard --logdir runs
```

Use `--mode rotate` for deterministic view regression. Omit `--use-depth` for
RGB-only training. Raw-image generation/logging currently assumes 128×128; latent
logging assumes 32×32, so arbitrary dataset resolutions need code changes.

Important CLI defaults (the example above overrides some of them):

| Option | Default |
| --- | --- |
| `--mode` | `generate` |
| `--dataset-path` | `datasets/small_512/` |
| `--epochs` / `--batch-size` | `1000` / `32` |
| `--lr` | `2e-4` |
| `--scheduler` | `cosine` (`step` and `none` also available) |
| `--warmup-epochs` / `--grad-clip` | `2` / `1.0` |
| `--seed` / `--validation-seed` | `42` / `0` |
| `--savepoint` | `50` epochs |
| `--rotation-encoding` | `6D` (currently the network always uses 6D) |
| `--logdir` | `runs` |

Use `python main.py --help` for the full argument list.

For latent training, `--use-encoding` reads the AE run name from the dataset root's
`encoding_metadata.json` and loads the autoencoder from
`<logdir>/encode/AE_RUN_NAME/`, preferring `final/model_checkpoint.pth`, then the
highest saved epoch, then a checkpoint in the run root. For example:

```bash
python main.py --mode generate --dataset-path datasets/demo \
  --use-encoding --run-name latent_flow
```

This still reads precomputed `.pt` latents referenced by `encoded_filename` in
the dataset CSV; generate those with the same autoencoder weights. The option is
available for `generate` and `rotate`. Raw-image runs
do not load an autoencoder. The selected checkpoint path is saved in run metadata.

Precompute encodings from the same AE run before latent training:

```bash
python data/generate_encoding.py --dataset-path datasets/demo --run-name my_autoencoder
```

Both splits are processed by default. Use `--split training` or `--split validation`
for one split, `--device cpu` to force CPU, and `--overwrite` to replace existing
latents. `--logdir` selects the runs root and `--output-dirname` selects the encoding
subdirectory within each scene. RGB-D input is inferred from the loaded weights.
Final checkpoints are preferred over the highest saved epoch.
Encoding generation writes one shared `encoding_metadata.json` containing
`run_name` at the dataset root; all encodings are assumed to use that AE run.

### Reproducible training and checkpoints

AE training adds low-frequency RGB noise in `AutoEncoder.train_step`: an 8×8
Gaussian field, upsampled with bilinear interpolation, with amplitude sampled
from [0, 0.2] on half the images. Input and reconstruction target use the same
corrupted image. RGB is clamped to [0, 1]; depth and validation remain unchanged.
These settings are hardcoded in the wrapper for this experiment.


Warmup increases the learning rate from 10% to 100% per batch over the initial
warmup epochs. Cosine decay runs over the remaining epochs; step scheduling starts
after warmup. With `--scheduler none`, warmup is followed by a constant learning
rate. Set `--grad-clip 0` or `--warmup-epochs 0` to disable those features.

Keep `--validation-seed` fixed when comparing training seeds. TensorBoard records
`train/grad_norm` (epoch mean before clipping), `train/grad_norm_max`, and
`train/learning_rate` (the last batch's learning rate), alongside training and
validation losses and sample panels. Seeding controls random draws but does not
guarantee bitwise-identical CUDA results across hardware/software versions.

Runs live under `runs/<mode>/<run-name>/`, with `metadata.json`, TensorBoard events,
`epoch_00050/model_checkpoint.pth`-style savepoints, and a
`final/model_checkpoint.pth`. Sample panels are logged to TensorBoard; standalone
PNG saving is currently commented out.

```bash
python main.py \
  --dataset-path datasets/scenes_10000_10_colored_spheres_depth \
  --mode generate --use-depth --epochs 100 \
  --resume-from rgbd_flow_seed42 --run-name rgbd_flow_seed42
```

Using the same resume/run name continues the run, restoring the completed epoch,
optimizer, scheduler when available, RNG state, clipping, and warmup settings.
`--epochs` is the total target epoch count. Using a different run name loads model
and optimizer state into a new run with fresh scheduling. Checkpoint selection
prefers `final/`, then the highest saved epoch, then a checkpoint in the run root.
Older checkpoints cannot restore RNG states they did not save and do not restart
warmup when continued.

## View a dataset or predicted RGB-D cloud

```bash
python -m viewer.view_dataset \
  datasets/scenes_10000_10_colored_spheres_depth 0 --split validation

python -m viewer.view_dataset \
  datasets/scenes_10000_10_colored_spheres_depth 0 --split validation \
  --run-name rgbd_flow_seed42 --steps 10
```

Drag the left mouse button to orbit. With a prediction run loaded, press **Space**
to predict the current camera view from the original dataset RGB-D image and append
its back-projected points. The source image and pose stay fixed for every request.
The camera orbits at a fixed radius; pan and zoom are absent.

Load a Gaussian scene alongside the point cloud with `--gaussian-run scene_0`
(resolved under `runs/gaussian/`) or a checkpoint file path. Press **Tab** to switch
between points and splats while preserving the camera. Both modes use a white
background. Space predicts additional views only in point-cloud mode. Gaussian
viewing requires CUDA and gsplat; no teacher is loaded. Splat rendering uses the
same 45-degree field of view as points, caps resolution at 512 pixels on the longest
edge, and refreshes when the camera or window size changes.

```bash
python -m viewer.view_dataset datasets/demo 0 --gaussian-run scene_0
python -m viewer.view_dataset --gaussian-run scene_0
```

Without a dataset, the viewer starts in splat mode; Tab shows the Gaussian centers
as colored points. When a dataset path is supplied, its index defaults to 0.

Prediction requires a `generate` or `rotate` run trained with `--use-depth` or
`--use-encoding`, plus its `metadata.json` and checkpoint. Latent runs load the
exact autoencoder checkpoint recorded in `encoding_checkpoint` in the run metadata,
encode the original source image, and decode the generated latent view. RGB-only
autoencoders use `--depth` as constant depth for generated points; RGB-D autoencoders
provide predicted depth. Inference supports
`--device auto`, `cpu`, or `cuda`; `auto` selects CUDA when available.

The default `--depth-source auto` currently requires saved ground-truth depth;
it does not fall back automatically. For an older RGB-only dataset, explicitly use
`--depth-source constant --depth 2.0`. This creates a temporary depth surface rather
than recovering geometry from the image. Appending predicted views does not yet
perform geometric fusion or enforce consistency between views.


## Code map

| Path | Responsibility |
| --- | --- |
| `main.py`, `trainer.py` | Training configuration, loop, validation, checkpoints |
| `models/RotationModel.py`, `models/Wrapper.py` | Attention network, flow/regression/AE objectives and sampling |
| `models/AutoEncoder.py`, `losses.py` | Autoencoder, MSE, optional LPIPS and latent regularization |
| `data/` | Procedural scenes, rendering, dataset loading, offline encodings |
| `logger.py` | Run metadata and TensorBoard output |
| `viewer/` | Camera orbit, RGB-D back-projection, point rendering, view prediction |
| `utils/` | Camera transforms, checkpoint resolution, RNG utilities |

## Gaussian scene model

`models.GaussianScene` stores a fixed set of trainable anisotropic 3D Gaussians
(position, log-scale, wxyz rotation, RGB color logits, and opacity logits).
It renders through [gsplat](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html).
Install the optional rendering backend in the project environment:

```bash
uv pip install gsplat==1.5.3
```

Rendering requires CUDA and a compatible CUDA build/toolchain for gsplat.
The backend is imported only when rendering, so other training modes and CPU
scene construction do not require it.

```python
from models import GaussianScene

scene = GaussianScene(num_gaussians=512).cuda()
result = scene.render(camera_pose, intrinsics, height=128, width=128)
rgb = result["rgb"]       # (B, 3, H, W)
depth = result["depth"]   # (B, 1, H, W), expected forward z-depth
alpha = result["alpha"]   # (B, 1, H, W)
```

Camera poses use the dataset's OpenGL camera-to-world convention; the model
converts them for gsplat internally. Intrinsics may be shared or batched.
Initialization uses random positions inside a sphere and direct RGB colors.
`save`/`load` handle state dicts (construct the same Gaussian count when loading).
`GaussianModel` in `models/Wrapper.py` provides pixel-space flow distillation;
`main.py` provides `--mode gaussian` with a camera-only training dataset.

```python
from models import GaussianModel

wrapper = GaussianModel(scene, frozen_flow_teacher)
optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=1e-3)
loss, metrics = wrapper.train_step(source_rgb, None, {
    "source": source_poses, "target": target_camera_poses, "intrinsics": intrinsics,
})  # source_rgb and camera tensors have a batch dimension
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The teacher must be a pixel or latent `FlowModel` using the same pixel range, camera
conventions, intrinsics and resolution as the source/render. It remains frozen
and in evaluation mode. `CameraDataset` owns the fixed source image, source pose,
and intrinsics, and includes them in each batch.
No target images or generated views are used. Pixel RGB-D teachers also
supervise expected z-depth through a separate distillation loss, weighted by 0.1.
Depth is not multiplied or weighted by alpha. RGB teachers use only RGB loss.
Latent teachers encode the source and rendered RGB or RGB-D image using their
recorded frozen autoencoder. From the noisy latent and teacher velocity they
estimate the clean latent as `z_t + (1-t)*velocity`, decode it, and fit rendered
RGB to the detached decoded RGB target with MSE. This RGB loss is added with unit
weight to the original latent distillation loss. The RGB term backpropagates
directly through the renderer; the latent term also propagates through the frozen
encoder through RGB. For latent RGB-D teachers, rendered depth is detached before
encoding: it informs the teacher but receives no direct gradient or separate
decoded-depth fitting loss. Both loss terms are logged. The diagnostic flag
`--detach-rendered-depth` tests this behavior with known-target latent fitting.
Teacher and autoencoder weights remain frozen and in evaluation mode.

With `x_t = (1-t)*noise + t*render`, the teacher's implied noise is
`x_t - t*v_teacher`. The injected gradient is its residual against sampled noise:
`t*(render - noise - v_teacher)`, using unit SDS weighting and a detached teacher
prediction. Loss reduction averages across batch, channels, and pixels. This is
an SDS gradient surrogate rather than a reconstruction loss or an evaluated KL.
Time is sampled uniformly from `[0.05, 0.95]` by default; bounds can be set in code.
Scene save/load uses the existing wrapper interface. Training checkpoints include
Gaussian weights and distillation settings; source conditioning comes from the dataset.

## Optimize a Gaussian scene using a flow teacher

First train a raw RGB or RGB-D `generate` run (no `--use-encoding`). Use
`--use-depth` for the teacher run to enable depth supervision during splatting;
the Gaussian run detects this from the teacher and loads source depth automatically.
Install the optional gsplat backend above, then:

```bash
python main.py --mode gaussian --teacher-run rgb_flow \
  --dataset-path datasets/demo \
  --batch-size 1 --epochs 50 --lr 0.001 \
  --savepoint 10 --run-name scene_0
```

`--teacher-run` resolves under `<logdir>/generate/`, preferring final then latest
epoch. Teacher metadata must identify an AttentionAutoEncoder flow run. Latent
teachers must record `encoding_checkpoint`; that exact autoencoder checkpoint
is loaded automatically, without passing `--use-encoding` to Gaussian training.
The dataset supplies only the chosen source image, pose, and intrinsics.
Gaussian settings are fixed in `main.py`: the first training image, 512 Gaussians,
camera radius 2.0, 128 training and 16 validation cameras per epoch, and time bounds
0.05–0.95.
Optimization samples spherical cameras on demand, looking at the origin;
no dataset target images supervise the scene. Keep camera radius/scene scale
consistent with the teacher's training distribution. One run optimizes one scene.

An epoch samples 128 fresh training cameras by default.
Gaussian training skips validation loss steps. A separate camera sampler supplies
16 cameras for validation image logging by default.
TensorBoard logs fixed-camera RGB, expected depth, opacity, and the source image.
Only Gaussian parameters are optimized and saved as model weights; teacher weights
are loaded separately. The teacher checkpoint path and Gaussian configuration are
recorded in `metadata.json`, along with the autoencoder checkpoint for latent teachers.

Resume with the same Gaussian configuration and teacher, plus
`--resume-from scene_0 --run-name scene_0`. Distillation settings, optimizer and RNG
state are restored for continuation. Use the same dataset for source conditioning.
As in other modes, `--epochs` is the total epoch count. Actual CUDA rasterization
and optimization quality still require GPU validation; CPU tests use a test renderer.
