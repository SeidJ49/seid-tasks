# Work Package 3: Motion-Aware Distillation Loss

## Objective
Implement knowledge distillation loss that masks distillation to dynamic Radar regions only.

## Main Goals
- Create motion-aware distillation loss function
- Prevent distillation in static regions where Radar is unreliable
- Support multi-stage training with selective teacher freezing
- Maintain compatibility with detection loss

## Key Tasks

### 3.1 Distillation Loss Implementation
- **File**: `opencood/loss/point_pillar_lite_bev_loss.py`
- **Loss Composition**:
  - L_det: Detection loss (standard PointPillar detection loss)
  - L_distill: Masked MSE only in dynamic Radar regions
  - L_distill = ||( F_R - F_L_teacher ) × M_dyn ||²

### 3.2 Masked Distillation Mechanism
- Use dynamic mask (M_dyn) from Doppler velocity
- Apply element-wise masking to feature difference
- Aggregate masked MSE loss
- Support both hard mask and soft-weighting variants

### 3.3 Stage-Specific Loss Configuration
- **Stage 1 (Teacher)**: No distillation, only L_det
- **Stage 2 (Radar Distill)**: Full distillation with high weight
- **Stage 3 (Fusion)**: Optional light distillation with reduced weight
- Teacher model frozen in Stage 2, optional in Stage 3

### 3.4 Loss Weighting & Logging
- Configurable weight for distillation loss
- Smooth spatial gate regularization: L_smooth = |∇_x W| + |∇_y W|
- Logging metrics:
  - det_loss
  - distill_loss (when applicable)
  - mean_gate (gate usage analysis)
  - radar_gate_usage (percentage of fusion from Radar)

### 3.5 Ablation Support
- YAML flags for selective loss components:
  - `use_distillation`: true/false
  - `use_reliability`: true/false
  - `use_gate_smooth`: true/false

## Deliverables
- [ ] `point_pillar_lite_bev_loss.py` with full loss implementation
- [ ] Distillation masking mechanism
- [ ] Loss weighting and aggregation logic
- [ ] Comprehensive logging for loss breakdown
- [ ] Loss function unit tests

## Success Criteria
- Distillation loss only computes gradients in dynamic regions
- Detection loss remains unaffected by distillation
- Loss values decrease monotonically during training
- Radar gate usage metrics are meaningful and trackable
- Support for all three training stages
