# Certified load-adaptive deepfake cascades (code for N20 manuscript)

Code accompanying the manuscript *Certified Load-Adaptive Cascades for Deepfake Detection on Edge–Cluster Systems under Generator Shift* (B. Sugiantoro, F. A. Hafizuddin; UIN Sunan Kalijaga Yogyakarta), submitted to Discover Computing.

## Contents
The scripts are in `code_N20.zip` (folders `hpc_scripts/` and `edge/`, plus `figs.py`); `figs.py` is also provided unzipped.
`results_N20.tar.gz` holds the per-draw results (`draws.csv.gz`) and summary tables (`summary.csv`) of every certification experiment (folders `face_main`, `face_main2`, `face_cc`, `face_cc2`, `face_cc_diff`, `face_group`, `face_strat`, `face_strat_v`, `face_strat_v10`, `pilot`, `pilot2`, `pilot3`) and the cluster benchmark `hpc_bench.json`. Columns of `draws.csv.gz`: condition, prevalence, split, label budget, draw, configuration, mode, release rate, risk, cost (GFLOPs), validity.
- `hpc_scripts/` Slurm job chain and Python scripts for the cluster node: data preparation, frozen-feature extraction, fine-tuning of tiers 1–2, certification draws, per-family and stratified runs, hardware benchmark.
- `edge/` Edge-device benchmark (`edge_bench.py`), launchers and measured output (`edge_results.json`).
- `figs.py` Draws the figures from the result tables.

## Protocol (summary)
- Data: DF40, DiFF, and real FaceForensics++/Celeb-DF frames released with DF40; 395,795 images after selection (max 6,000 per generator).
- Shift defined at generator-family level before any result was seen.
- 10 random splits; prevalence 50%, 10%, 1%; label budgets n ∈ {800, 3200, 6400, 12800}; alpha = 0.05; 3 tiers (MobileNetV3-L, ResNet-50, frozen CLIP ViT-L/14 + DINOv2 ViT-L/14).

## Hardware
Cluster node: 2× Intel Xeon 6530P, 500 GB RAM, 2× NVIDIA L40S, Slurm, Apptainer 1.5.0, NGC PyTorch 25.08. Edge device: Intel Core i7-1260P, 32 GB RAM, NVIDIA RTX 3050 Laptop 4 GB, Windows 11.
Paths in the scripts (`/virtiofs/work/bambang/cascade`) refer to the authors' cluster and must be adapted.

## Data and licences
The image datasets are not included because their licences do not allow redistribution (DiFF: request from its authors; DF40, FaceForensics++, Celeb-DF: from their providers). Per-image scores are available from the corresponding author on request.

## Licence
Code: MIT. See `LICENSE`. Contact: bambang.sugiantoro@uin-suka.ac.id
