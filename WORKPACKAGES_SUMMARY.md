# Lite-BEV Thesis: Six Work Packages (Complete Scope)

## Overview
Six interconnected work packages for implementing a lightweight LiDAR-Radar BEV fusion model with motion-aware distillation and reliability-guided gating for adverse weather robustness.

## Complete Work Package Breakdown

| WP | Title (German) | Main Goal | Stage | Status |
|---|---|---|---|---|
| **WP1** | Baseline Setup | Create stable baselines (LiDAR, Radar, Naive Fusion) | 1 | Ready |
| **WP2** | LiDAR Teacher Model | Prepare frozen LiDAR teacher on clear weather | 1 | Ready |
| **WP3** | Motion-Aware Radar Distillation | Train radar with Doppler-masked distillation | 2 | Ready |
| **WP4** | LiDAR Reliability Estimation | Create heuristic reliability map (point count + occupancy) | 3 | Ready |
| **WP5** | Reliability-Guided Gated Fusion | Implement spatial gate for adaptive fusion | 3 | Ready |
| **WP6** | Training Strategy | Implement 3-stage training pipeline | 3 | Ready |

## Implementation Timeline

```
Stage 1: Foundation (WP1 + WP2)
├─ WP1: Build three independent PointPillar baselines
│  ├─ LiDAR-only
│  ├─ Radar-only
│  └─ Naive Fusion (concat)
│
└─ WP2: Train LiDAR teacher on clear weather, freeze for distillation
   └─ Checkpoint: lidar_teacher_clear_weather.pth
   
   ↓ (use frozen teacher)
   
Stage 2: Radar Improvement (WP3)
├─ WP3: Motion-Aware Distillation
│  ├─ Extract Doppler velocity from radar
│  ├─ Build dynamic mask (|v_r| > 0.5 m/s)
│  ├─ Implement masked MSE loss
│  └─ Train radar with frozen teacher supervision
│     └─ Checkpoint: radar_distilled.pth
│
   ↓ (use frozen teacher + distilled radar)
   
Stage 3: Fusion with Intelligence (WP4 + WP5 + WP6)
├─ WP4: LiDAR Reliability Estimation
│  ├─ Combine: Point Count + Occupancy Sparsity
│  ├─ Create: U_L ∈ [0,1] reliability map
│  └─ Validate: Correlates with weather degradation
│
├─ WP5: Reliability-Guided Gated Fusion
│  ├─ Gate: W = σ(φ([F_L, F_R, U_L]))
│  ├─ Fusion: F_fused = (1-W)⊙F_L + W⊙F_R
│  ├─ Output: Gate heatmaps + statistics
│  └─ Ablations: with/without reliability map
│
└─ WP6: Three-Stage Training Pipeline
   ├─ Stage 1: Train LiDAR teacher on clear weather
   ├─ Stage 2: Train radar with distillation (teacher frozen)
   └─ Stage 3: Train full fusion on mixed weather
      └─ Checkpoint: fusion_model_final.pth
```

## Detailed Work Package Goals

### WP1: Baseline Setup (Stage 1)
**Goal**: Establish stable, reproducible baselines before adding any method components.

**Deliverables**:
- Three trained PointPillar detectors: LiDAR-only, Radar-only, Naive Fusion
- Config files for all three
- BEV feature tensor shapes documented: (B, C, H, W)
- Results table: mAP per weather condition (Clear, Fog light/heavy, Rain light/heavy)
- Inference speed (FPS) for all models

**Success Criteria**:
- All models run through complete epoch without errors
- Results reproducible (±1% variance between runs)
- Performance ranking: LiDAR > Naive Fusion > Radar (expected)

---

### WP2: LiDAR Teacher Model (Stage 1)
**Goal**: Prepare a strong, frozen LiDAR model on clear weather for distillation.

**Deliverables**:
- Trained LiDAR-only PointPillar on Clear-Weather data only
- Frozen model: `requires_grad = False`, `eval()`
- Documentation: Which BEV layer used as distillation target
- Feature shape and extraction code

**Success Criteria**:
- Teacher achieves strong mAP on clear weather
- Teacher frozen and cannot be updated
- Checkpoint saveable and loadable independently

---

### WP3: Motion-Aware Radar Distillation (Stage 2)
**Goal**: Train radar encoder with Doppler-masked supervision from frozen teacher.

**Deliverables**:
- Doppler velocity extraction from 4D radar
- Dynamic mask generation: binary BEV mask for |v_r| > 0.5 m/s
- Masked MSE loss: `||( F_R - F_L_teacher ) × M_dyn ||²`
- Trained radar-only model with distillation

**Success Criteria**:
- Dynamic mask correctly identifies moving objects
- Distillation loss decreases during training
- Radar mAP improves: +2-5% on clear weather vs. non-distilled
- Radar features qualitatively similar to LiDAR in dynamic regions

---

### WP4: LiDAR Reliability Estimation (Stage 3)
**Goal**: Create heuristic reliability map showing where LiDAR fails under degraded weather.

**Deliverables**:
- Reliability map: `U_L ∈ [0,1]^(H×W)`
- Heuristic: Combine Point Count + Occupancy Sparsity
- NOT: Pure feature variance (insufficient)
- Visualizations: Heatmaps for clear/fog/rain
- Correlation analysis: U_L vs. mAP degradation (negative expected)

**Success Criteria**:
- U_L varies meaningfully between weather conditions
- Clear weather: U_L mostly low (reliable)
- Heavy rain: U_L mostly high (unreliable)
- Correlates with detection performance drop

---

### WP5: Reliability-Guided Gated Fusion (Stage 3)
**Goal**: Implement lightweight spatial gate for adaptive LiDAR-Radar fusion.

**Deliverables**:
- Spatial Gate (not channel-wise): `W ∈ [0,1]^(H×W)`
- Gate formula: `W = σ(φ([F_L, F_R, U_L]))`
- Fusion: `F_fused = (1-W)⊙F_L + W⊙F_R`
- Gate heatmaps and usage statistics
- Ablations: Simple concat vs. gate without reliability vs. full gate

**Success Criteria**:
- Gate values in [0,1] range (no collapse to 0 or 1)
- Mean(W) increases with weather degradation
- Ablation shows: Full gate > gate without reliability > simple concat
- Gate is interpretable and visualizable

---

### WP6: Training Strategy (Stage 3)
**Goal**: Implement complete 3-stage training pipeline integrating all components.

**Deliverables**:
- Stage 1: Train LiDAR teacher on clear weather
- Stage 2: Train distilled radar (teacher frozen)
- Stage 3: Train full fusion on mixed weather
- Configuration files for all stages
- Training logs and checkpoints

**Success Criteria**:
- All three stages train without errors
- Stage transitions: Teacher → Distilled Radar → Full Fusion
- Final model outperforms all baselines on adverse weather
- Checkpoints and logs are organized and reproducible

---

## Required Baselines & Ablations

### Baselines (from WP1)
- [ ] LiDAR-only
- [ ] Radar-only (non-distilled)
- [ ] Naive LiDAR-Radar fusion (concat/sum)

### With Distillation (from WP3)
- [ ] Radar-only with motion-aware distillation

### With Gating (from WP5)
- [ ] Simple concat fusion (no gate)
- [ ] Spatial gate WITHOUT reliability map
- [ ] Spatial gate WITH reliability map (full method)

### Gate Regularization (conditional, if needed)
- [ ] Gate without smoothness regularization
- [ ] Gate with smoothness regularization (if collapses)

**Expected Performance Ranking**:
```
Full Lite-BEV > Gate+Reliability > Gate (no reliability) > Radar (distilled) > Naive Fusion > Radar (non-distilled)
```

---

## Key Architectural Decisions

### Dataset & Data
- Doppler velocity threshold: `|v_r| > 0.5 m/s` for dynamic regions
- BEV pillar-based aggregation
- Clear-weather used for Stage 1+2, mixed-weather for Stage 3

### Reliability Map Heuristics (NOT pure variance)
- **Point Count**: Number of LiDAR points per pillar
- **Occupancy Sparsity**: Proportion of occupied pillars in 3×3 neighborhood
- **Interpretation**: High U_L = low reliability (schlechtes Wetter)

### Gate Design (Spatial only, first version)
- **Inputs**: Concatenated [F_L, F_R, U_L]
- **Architecture**: 1×1 Conv(C→C/4) + ReLU + Conv(C/4→1) + Sigmoid
- **Output**: Scalar weight per BEV cell in [0,1]
- **NO**: Channel-wise gating (added later if needed)

### Training Strategy (Staged, NOT joint)
1. **Stage 1**: LiDAR teacher on clear weather
2. **Stage 2**: Radar with motion-aware distillation (teacher frozen)
3. **Stage 3**: Full fusion on mixed weather (optional: continue light distillation)

---

## Evaluation Requirements

### Weather Conditions
- Clear
- Fog (light and heavy)
- Rain (light and heavy)

### Metrics
- AP / mAP (primary)
- Class-wise metrics (if applicable)
- FPS (for lightweight claim)
- Parameter count (if applicable)

### Qualitative Analysis
- Motion mask visualizations
- Reliability map heatmaps
- Gate heatmaps per weather condition
- BEV feature visualizations (before/after distillation)
- Weather-dependent gate behavior analysis

---

## Repository Structure

```
seid-tasks/
├── docs/workpackages/
│   ├── WP1_Baseline_Setup.md              ← Baseline models
│   ├── WP2_LiDAR_Teacher.md               ← Teacher preparation
│   ├── WP3_Motion_Aware_Distillation.md   ← Radar training with distillation
│   ├── WP4_Reliability_Estimation.md      ← Reliability heuristics
│   ├── WP5_Gated_Fusion.md                ← Spatial gate implementation
│   └── WP6_Training_Strategy.md           ← 3-stage training pipeline
├── .gitlab/issue_templates/
│   └── workpackage.md
└── WORKPACKAGES_SUMMARY.md                ← This file
```

## GitLab Issues

All 6 work packages are tracked as GitLab issues:
- **Issue #1**: WP1 - Baseline Setup
- **Issue #2**: WP2 - LiDAR Teacher Model
- **Issue #3**: WP3 - Motion-Aware Radar Distillation
- **Issue #4**: WP4 - LiDAR Reliability Estimation
- **Issue #5**: WP5 - Reliability-Guided Gated Fusion
- **Issue #6**: WP6 - Training Strategy

---

## Implementation Tips

1. **Follow the sequence**: WP1 → WP2 → WP3 → WP4 → WP5 → WP6
2. **Don't jump ahead**: Each WP depends on previous checkpoints
3. **Document as you go**: Save configs, logs, and results
4. **Test early, ablate often**: Validate each component before combining
5. **Keep it simple first**: No fancy regularization unless gate collapses

---

## Success Criteria for Thesis

- ✓ All 6 WP completed with deliverables
- ✓ Baselines and ablations show expected ranking
- ✓ Final method (Lite-BEV) outperforms single-sensor on adverse weather
- ✓ Inference remains lightweight (>10 Hz expected)
- ✓ All results reproducible with documented hyperparameters
- ✓ Qualitative analysis explains WHEN and WHERE the method works

This completes the **full scope of the Lite-BEV thesis implementation**.
