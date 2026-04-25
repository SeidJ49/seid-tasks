# Lite-BEV Thesis: Three Core Work Packages

## Overview
Three interconnected work packages for implementing a lightweight LiDAR-Radar BEV fusion model for autonomous driving perception in adverse weather conditions.

## Work Package Breakdown

| WP | Title (German) | Main Goal | Status |
|---|---|---|---|
| **WP1** | Baseline Setup | Stable, reproducible baselines without fusion or distillation | Ready |
| **WP2** | LiDAR Teacher Model | Prepare frozen LiDAR teacher for distillation supervision | Ready |
| **WP3** | Motion-Aware Radar Distillation | Train radar encoder with motion-masked distillation | Ready |

## Implementation Sequence

```
WP1: Baseline Setup
    └─ Build three independent PointPillar detectors
       ├─ LiDAR-only
       ├─ Radar-only  
       └─ Naive Fusion (simple concat)
       
        ↓ (use LiDAR results as baseline)
        
WP2: LiDAR Teacher Model
    └─ Take best LiDAR-only from WP1
    └─ Train on Clear-Weather only
    └─ Freeze for use in distillation
    
        ↓ (freeze teacher)
        
WP3: Motion-Aware Radar Distillation
    └─ Extract Doppler velocity from radar points
    └─ Build dynamic mask (M_dyn) for moving objects
    └─ Implement masked MSE loss
    └─ Train radar encoder with distillation from frozen teacher
```

## Detailed Goals per Work Package

### WP1: Baseline Setup (Detaillierte Beschreibung)

**Goal**: Create stable, reproducible baselines without any fusion or distillation.

**Deliverables**:
- Three trained PointPillar models:
  1. **LiDAR-only**: Input only LiDAR points → 3D bounding boxes
  2. **Radar-only**: Input only Radar points → 3D bounding boxes
  3. **Naive Fusion**: LiDAR + Radar with simple concat (no gating) → 3D bounding boxes

**Key Tasks**:
- Create config files: `cfgs/lidar_only_pointpillars.yaml`, `cfgs/radar_only_pointpillars.yaml`, `cfgs/naive_fusion_pointpillars.yaml`
- Implement three model classes in `models/`
- Adapt data loaders to handle both sensors
- Document BEV feature tensor shapes and spatial resolution
- Train all three models on identical dataset with same hyperparameters
- Evaluate on multiple weather conditions: Clear, Fog (light/heavy), Rain (light/heavy)

**Success Metrics**:
- All three models run without errors through complete epoch
- Results are reproducible (±1% variance between runs)
- BEV tensor shapes documented: `(B, C, H, W)`
- Results table completed (mAP per weather condition + FPS)

**What is NOT included**:
- No gating, no reliability maps
- No distillation, no doppler masking
- No frozen teacher, no staged training
- No special loss functions

### WP2: LiDAR Teacher Model (Detaillierte Beschreibung)

**Goal**: Prepare a strong, frozen LiDAR model trained on clear weather that serves as ground truth for radar distillation.

**Deliverables**:
- One trained, frozen LiDAR-only PointPillar model
- Clear identification of BEV feature level for distillation

**Key Tasks**:
- Take best LiDAR-only model from WP1
- **Retrain on Clear-Weather data only** (not mixed weather)
- Freeze all parameters: `requires_grad = False`, `eval()`
- Select specific layer as teacher target (recommended: features after PointPillarScatter)
- Document feature level:
  - Variable name in code
  - Shape: `(B, C, H, W)` with specific C, H, W values
  - How to extract in forward pass
- Save teacher checkpoint: `checkpoints/lidar_teacher_clear_weather.pth`
- Validate performance on Clear/Fog/Rain splits

**Success Metrics**:
- Teacher model frozen and cannot be updated
- Clear documentation of which layer is used as distillation target
- Checkpoint can be loaded in separate script
- Validation metrics on Clear, Fog, Rain splits documented

**What is NOT included**:
- No distillation (that's WP3)
- No radar training
- No fusion, no gating, no reliability maps
- No architecture changes

### WP3: Motion-Aware Radar Distillation (Detaillierte Vorbereitung)

**Goal**: Train radar encoder using frozen teacher knowledge, masked only to dynamic regions by Doppler velocity.

**Deliverables**:
- Doppler-based dynamic mask generation function
- Custom masked distillation loss implementation
- One trained distilled radar-only model checkpoint

**Key Tasks**:
- Extract Doppler (radial velocity) from radar points: `v_r`
- Build dynamic mask algorithm:
  1. Project radar points to BEV coordinates
  2. For each pillar: if any point has `|v_r| > 0.5 m/s`, mark as dynamic
  3. Optional: dilate mask (3×3) to capture object surroundings
- Implement masked MSE loss: `L_distill = ||( F_R - F_L_teacher ) × M_dyn ||²`
- Integrate into training loop:
  - Total loss = `L_detection + lambda * L_distill` (start with `lambda = 0.1`)
- Train radar-only model on Clear-Weather with distillation
- Create qualitative visualizations showing radar BEV features before/after distillation

**Success Metrics**:
- Dynamic mask correctly identifies moving objects
- Distillation loss decreases during training
- Radar model shows improvement over non-distilled baseline (expect +2-5% mAP on Clear)
- Radar BEV features visually similar to LiDAR features in dynamic regions
- Performance improves on Fog/Rain due to better geometric priors

**What is NOT included**:
- No fusion with LiDAR
- No reliability map (WP4 in extended version)
- No gating mechanism
- No training on mixed weather (Clear only)
- No modifications to frozen teacher

## Key Design Decisions

**Dataset**:
- Doppler velocity threshold: `|v_r| > 0.5 m/s` for dynamic regions
- BEV pillar-based aggregation for dynamic mask

**Architecture**:
- Base: PointPillar encoder (VFE + Scatter + Backbone)
- Three independent training tracks in sequence
- Teacher model frozen after WP2

**Training**:
- WP1: Train baselines on identical clean dataset
- WP2: Retrain LiDAR on Clear-Weather subset, then freeze
- WP3: Train Radar with masked distillation from frozen teacher

**Evaluation**:
- Multiple weather conditions: Clear, Fog (light/heavy), Rain (light/heavy)
- Metrics: mAP (primary), NDS (if applicable), FPS (for "lightweight" claim)
- Qualitative: BEV feature heatmaps, mask visualizations

## Repository Structure

```
seid-tasks/
├── docs/workpackages/
│   ├── WP1_Dataset.md                    ← Baseline Setup details
│   ├── WP2_Model_Core.md                 ← Teacher Model details
│   └── WP3_Distillation_Loss.md          ← Motion-Aware Distillation details
├── .gitlab/
│   └── issue_templates/
│       └── workpackage.md                ← Issue template
└── WORKPACKAGES_SUMMARY.md               ← This file
```

## GitLab Issues

All work packages have been created as GitLab issues:
- **Issue #1**: WP1 - Baseline Setup
- **Issue #2**: WP2 - LiDAR Teacher Model
- **Issue #3**: WP3 - Motion-Aware Radar Distillation

Closed non-relevant issues:
- Issues #4-7 (originally placeholder issues)

## Next Steps

1. Review each issue and detailed documentation in `docs/workpackages/`
2. Start with **WP1**: Build three baseline models
3. Once baselines are stable, proceed to **WP2**: Prepare teacher
4. Finally, implement **WP3**: Motion-aware distillation
5. Track progress in GitLab by updating issue status

## Extended Roadmap (Future Work)

Future work packages (beyond current scope) could include:
- **WP4**: LiDAR Reliability Estimation (heuristic or learned)
- **WP5**: Reliability-Guided Gated Fusion (gate mechanism + combined training)
- **WP6**: Inference & Deployment (production-ready pipeline)
- **WP7**: Automation & Continuous Evaluation (automated experiments)

These would build upon the foundation of WP1-WP3.
