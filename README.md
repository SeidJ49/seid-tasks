# Lite-BEV: Lightweight LiDAR-Radar BEV Fusion

This repository manages the tasks and documentation for the Master's Thesis: **"Lite-BEV: Lightweight LiDAR-Radar BEV Fusion for Robust 3D Object Detection in Adverse Weather"**.

## Project Wiki
Detailed documentation, work packages, and meeting notes can be found in the project Wiki:
👉 **[Go to Wiki](./wiki/home.md)** (or use the GitLab Wiki tab)

---

## Thesis Overview
This thesis investigates multi-modal sensor fusion methods combining **LiDAR and 4D Radar** for 3D object detection in autonomous driving, specifically focusing on performance under adverse weather conditions (rain, fog). 

To address the limitations of LiDAR in poor visibility, this work proposes **Lite-BEV**, a lightweight Bird’s-Eye View (BEV) fusion framework. The approach avoids computationally expensive Transformer-based cross-attention and instead leverages two core mechanisms:

1.  **Doppler-Masked Geometric Distillation:** A training strategy where a 4D Radar encoder learns LiDAR-quality geometric features, specifically for dynamic objects identified via Radar's radial velocity.
2.  **Aleatoric Uncertainty-Driven Gating:** An inference-time fusion gate that dynamically switches between LiDAR and Radar based on estimated LiDAR reliability (e.g., spatial variance or point density drops in fog/rain).

The method is built upon the **OpenPCDet** framework, using **PointPillars** as a baseline, and is evaluated on the **MANtruckscenes** dataset.

---

## Repository Structure
- `docs/workpackages/`: Detailed technical descriptions of each work package.
- `wiki/`: Project documentation and progress logs (synced as a submodule).
- `.gitlab/issue_templates/`: Templates for creating work-package related issues.

## Getting Started
To initialize the wiki submodule:
```bash
git submodule update --init --recursive
```
