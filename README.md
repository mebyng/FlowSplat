# Project: Flowsplat

## Flow-Matched Image-to-3D Gaussian Splatting (Low-VRAM Generative Geometry System)

---

## Overview

FlowSplat is a lightweight experimental system that combines:

- Flow matching for **novel view synthesis in image space**
- Flow matching for **3D Gaussian scene generation**
- Cross-space consistency between rendered 3D and predicted image motion

It is designed to run on a consumer GPU (~1.5 GB VRAM) using only synthetic data and small neural networks.

No pretrained models are required.

---

## Core Idea

FlowSplat unifies two flow processes:

### 1. Image-space flow (view synthesis)

Learn how images change under camera motion:

Image A + direction (A → B) → Image B

Instead of diffusion, we learn a velocity field that transports one view into another.

---

### 2. 3D Gaussian flow (scene formation)

Learn how a noisy point cloud becomes a structured object:

Random Gaussians → Structured 3D object

Each scene is represented as a fixed set of Gaussians:

(x, y, z, r, g, b, σ, α)

---

### 3. Cross-domain bridge

Ensure consistency between 2D predictions and 3D reconstruction:

Render(Gaussians) ≈ Predicted novel views

---

## System Pipeline

Synthetic Data → Image Flow Model → Gaussian Scene Model → Consistency Alignment → 3D Output

---

## Stage 0 — Synthetic Data Generation

Generate training data using procedural geometry:

- Primitive shapes (cubes, spheres, cylinders)
- Random lighting conditions
- Camera orbits around objects
- Low resolution renders (64×64 or 128×128)

Output:

(object, camera poses) → image sequences

No real-world datasets are used.

---

## Stage 1 — Image Flow Model (View Synthesis)

### Goal

Predict how an image changes under camera motion.

### Inputs

- Image A
- direction update (A → B)

### Output

- Image B

---

### Flow Matching formulation

Interpolation:

xt = (1 − t) * xA + t * xB

Velocity target:

v = xB − xA

Training objective:

L_flow = || vθ(xt, t, camera_condition) − v ||²

---

### Model

- Small U-Net or lightweight transformer
- ~2–5M parameters
- Input resolution: 64×64

---

## Stage 2 — Gaussian Scene Representation

A scene is represented as N Gaussians:

G = { (x, y, z, r, g, b, σ, α) } for i = 1..N

Typical values:
- N = 256 (minimal)
- N = 512 (balanced)
- N = 1024 (higher quality)

Initialization:
- random noise OR
- sparse geometric initialization

---

## Stage 3 — Gaussian Flow Matching

Learn a transformation from noise to structure:

G0 (noise) → G1 (object)

Flow objective:

vθ(G, t) ≈ G1 − G0

This operates directly in 3D parameter space.

---

## Stage 4 — Image ↔ 3D Consistency Bridge

Render Gaussian scene:

Render(G, camera A) → IA_hat  
Render(G, camera B) → IB_hat  

Use image flow model:

IA → predicted flow → IB_pred

Consistency constraint:

IB_hat ≈ IB_pred

This aligns geometry with learned view dynamics.

---

## Loss Functions

### 1. Image flow loss

L_flow = || vθ(xt, t) − (xB − xA) ||²

### 2. Rendering loss

L_render = || Render(G) − I ||²

### 3. Consistency loss

L_consistency = || Render(G, B) − Flow(A → B) ||²

---

## Training Strategy

### Phase 1 — Image Flow Only
Train view synthesis model on synthetic image pairs.

### Phase 2 — Gaussian Optimization
Fit Gaussians to multi-view renders.

### Phase 3 — Consistency Training
Align Gaussian rendering with image-space predictions.

### Phase 4 (optional) — Joint training
End-to-end optimization:

image → flow → gaussians → render → losses → backprop

---

## Implementation Constraints

Designed for low VRAM:

- Resolution: 64×64 or 128×128
- Batch size: 1–4
- Gaussians: ≤ 1000 per scene
- Model size: ≤ 5M parameters
- No pretrained networks

---

## Outputs

FlowSplat produces:

- Novel view synthesis from single or sparse images
- 3D Gaussian splat reconstructions
- Real-time interactive rendering
- Consistent geometry across viewpoints

---

## Extensions

- Learned camera trajectory generation
- Adaptive Gaussian splitting and pruning
- Category conditioning (chair, mug, etc.)
- Higher resolution rendering
- Dynamic scenes with time-varying Gaussians
- End-to-end joint flow training

---

## End Goal

Single image → learned view flow → reconstructed Gaussian splat → interactive 3D scene

FlowSplat is a minimal but structurally modern generative 3D system inspired by current flow-matching and Gaussian splatting research, designed to be fully trainable on consumer hardware.