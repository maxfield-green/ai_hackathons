# AI Hackathons

Hands-on notebooks for running and training AI models on scientific problems.

---

## Notebook: Weather Downscaling with CorrDiff

**`notebooks/corrdiff_demo_for_hackathons.ipynb`**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxfield-green/ai_hackathons/blob/main/notebooks/corrdiff_demo_for_hackathons.ipynb)

Train and run a generative AI model that sharpens coarse weather forecasts into high-resolution predictions — no prior ML experience required.

### What you will do

1. Train a neural network that learns to convert low-resolution (~25 km) weather data into high-resolution (~3 km) predictions.
2. Run the trained model on a sample and compare its output against ground truth.
3. Apply the model to a new geographic region using real-time global weather data (GFS).

### Background: what is downscaling?

Global weather models cover the entire Earth but at coarse resolution — grid cells are roughly 25–30 km wide. Many real-world decisions depend on finer details: how wind behaves around mountains, where rain concentrates in a valley, or how temperatures vary across a city. **Downscaling** is the process of using AI to fill in that fine-scale detail from the coarse model output.

**CorrDiff** (Corrective Diffusion Model) is an NVIDIA research model built for this task. It uses a two-stage approach:
- A **regression model** learns a direct mapping from coarse to fine (deterministic).
- A **diffusion model** adds realistic variability around that estimate (stochastic).

This notebook covers the regression stage, which is faster to train and still demonstrates the full pipeline end to end.

### Dataset

The notebook uses **HRRR-Mini**, a compact dataset derived from NOAA's High-Resolution Rapid Refresh (HRRR) model over the continental United States. It includes:

| Group | Contents | Resolution |
|-------|----------|------------|
| `input` | Low-resolution predictor fields | 8×8 (~24 km) |
| `output` | High-resolution target fields | 64×64 (~3 km) |
| `invariant` | Static fields (elevation, land-sea mask) | 64×64 (~3 km) |

The dataset is downloaded automatically from NVIDIA GPU Cloud (NGC) during the notebook (~2 GB).

### Notebook sections

| Section | What happens |
|---------|-------------|
| **0. Prerequisites** | Detect GPU or fall back to CPU |
| **1. Install dependencies** | Clone PhysicsNeMo and install packages |
| **2. Download dataset** | Fetch HRRR-Mini from NGC |
| **3. Explore the dataset** | Inspect data structure and visualize samples |
| **4. Train the model** | Train a regression UNet on HRRR-Mini |
| **5. Run inference** | Generate predictions from the trained checkpoint |
| **6. Visualize results** | Compare prediction vs truth; compute error metrics |
| **7. Inspect model inputs** | Plot all input channels fed to the model |
| **8. Regional transfer** | Download live GFS data, run model over a new region |
| **9. Summary** | Recap and suggested next experiments |

### Requirements

- **Google Colab** (recommended) — the notebook is designed to run there without any local setup.
- **GPU runtime** strongly recommended (T4 or better). CPU fallback is supported but training will be very short.
- No local installations needed when running in Colab.

### What to expect

Training runs for a small number of steps by design (2 000 images on GPU, 2 on CPU), so predictions will not look physically perfect — the goal is to walk through the full pipeline end to end. Results improve substantially with longer training.

The regional transfer experiment in Section 8 downloads real-time GFS data from NOAA and applies the HRRR-trained model to a user-selected 192 km × 192 km domain anywhere on the globe. This is a domain-shift experiment: useful for exploring the workflow, but not a calibrated operational forecast.

### References

- [CorrDiff paper — arXiv 2309.15214](https://arxiv.org/abs/2309.15214)
- [PhysicsNeMo GitHub](https://github.com/NVIDIA/physicsnemo)
- [NOAA NOMADS (GFS data)](https://nomads.ncep.noaa.gov/)
