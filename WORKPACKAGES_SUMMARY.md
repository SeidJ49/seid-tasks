# Lite-BEV Implementation Work Packages

## Overview
Seven interconnected work packages for implementing a lightweight LiDAR-Radar BEV fusion model for autonomous driving perception.

## Work Package Breakdown

| WP | Title | Main Goal | Complexity |
|---|---|---|---|
| **WP1** | Dataset Setup & Preparation | Create LiDAR-Radar fusion dataset with Doppler-based dynamic masks | High |
| **WP2** | Lite-BEV Model Core | Implement spatial gate architecture with reliability heuristic | High |
| **WP3** | Motion-Aware Distillation Loss | Create masked distillation loss for dynamic Radar regions | Medium |
| **WP4** | Multi-Stage Training Pipeline | Implement 3-stage training (Teacher → Distill → Fusion) | High |
| **WP5** | Baseline & Ablation Experiments | Set up 6 baseline variants for component validation | Medium |
| **WP6** | Inference & Deployment | Create production-ready inference and evaluation tools | Medium |
| **WP7** | Automation & Continuous Evaluation | Implement automated experiment framework | High |

## Dependency Graph

```
WP1 (Dataset)
  ↓
WP2 (Model) ← WP3 (Loss)
  ↓           ↓
WP4 (Training)
  ↓
WP5 (Ablation) → WP6 (Inference)
  ↓
WP7 (Automation)
```

## Implementation Sequence

### Phase 1: Foundation (WP1 + WP2)
- Create dataset infrastructure
- Build core model architecture
- Establish training compatibility

### Phase 2: Training (WP3 + WP4)
- Implement distillation loss
- Set up multi-stage training pipeline
- Verify stage transitions

### Phase 3: Validation (WP5 + WP6)
- Run ablation experiments
- Validate component contributions
- Optimize inference

### Phase 4: Automation (WP7)
- Automate all experiments
- Generate thesis-ready reports
- Ensure reproducibility

## Key Metrics & Success Criteria

### Per Work Package
- **WP1**: Dataset produces valid tensors with meaningful dynamic masks
- **WP2**: Forward pass outputs PSM/RM with gate values in [0,1]
- **WP3**: Distillation loss decreases during Stage 2 training
- **WP4**: All three stages complete without errors
- **WP5**: Clear performance improvement: naive < baselines < full Lite-BEV
- **WP6**: Inference >10 Hz, compatible with embedded systems
- **WP7**: All experiments reproducible with documented results

## GitLab Issues
All work packages have been created as GitLab issues in the repository:
- Issue #1: WP1 - Dataset
- Issue #2: WP2 - Model Core
- Issue #3: WP3 - Distillation Loss
- Issue #4: WP4 - Training Pipeline
- Issue #5: WP5 - Ablation Experiments
- Issue #6: WP6 - Inference & Deployment
- Issue #7: WP7 - Automation

## Documentation
Detailed specifications for each work package are in `docs/workpackages/`:
- `WP1_Dataset.md`
- `WP2_Model_Core.md`
- `WP3_Distillation_Loss.md`
- `WP4_Training_Pipeline.md`
- `WP5_Ablation.md`
- `WP6_Inference.md`
- `WP7_Automation.md`

## Main Goals at a Glance

### 1. Lightweight Fusion Architecture
Implement BEV-level fusion without heavy Transformer cross-attention for real-time inference.

### 2. Motion-Aware Learning
Use Radar velocity information to guide distillation only in dynamic regions where Radar is reliable.

### 3. Spatial Gating
Learn adaptive sensor weighting through spatial gate mechanism conditioned on reliability maps.

### 4. Three-Stage Training
Separate training for robustness: Teacher pretraining → Radar branch learning → Full fusion fine-tuning.

### 5. Systematic Validation
Comprehensive ablation studies to validate each component's contribution.

### 6. Production Readiness
Efficient inference pipeline suitable for real-time autonomous driving systems.

### 7. Reproducible Research
Fully automated experiment framework for thesis validation and future extension.
