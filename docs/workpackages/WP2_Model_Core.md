# Work Package 2: Lite-BEV Model Core Implementation

## Objective
Implement lightweight BEV-based LiDAR-Radar fusion model with spatial gating and reliability-guided fusion.

## Main Goals
- Create efficient LiDAR-Radar BEV fusion model without heavy Transformer cross-attention
- Implement spatial gate mechanism for adaptive sensor weighting
- Design simple inference pipeline for production use
- Support three-stage training: Teacher → Distillation → Fusion

## Key Tasks

### 2.1 Core Lite-BEV Model Architecture
- **File**: `opencood/models/point_pillar_lite_bev_lidar_radar.py`
- **Architecture**:
  1. **LiDAR Encoder**: PointPillar VFE → Scatter → Backbone → F_L
  2. **Radar Encoder**: PointPillar VFE → Scatter → Backbone → F_R
  3. **Reliability Map U_L**: Heuristic from LiDAR (point density, intensity)
  4. **Spatial Gate**: W = sigmoid(φ([F_L, F_R, U_L]))
  5. **Fusion**: F_fused = (1-W)·F_L + W·F_R
  6. **Detection Head**: Shared head outputting PSM and RM (optional DM)

### 2.2 Reliability Heuristic (U_L)
- Per-LiDAR-pillar metrics:
  - Point count per pillar
  - Local density in neighborhood
  - Intensity values (if available)
- Interpretation: High U_L = low LiDAR reliability
- Normalize to [0, 1]

### 2.3 Spatial Gate Design
- Simple convolution-based gate block φ:
  - Conv(2C+1 → C/4, 1×1) + ReLU
  - Conv(C/4 → 1, 1×1) + Sigmoid
- Output gate W: [B, 1, H, W] ∈ [0, 1]
- Optional smooth regularization: L_smooth = |∇_x W| + |∇_y W|

### 2.4 Output Compatibility
- **Required**: PSM (objectness map), RM (regression map)
- **Optional**: DM (displacement map), gate_map, reliability_map for analysis
- Maintain compatibility with standard detection loss and postprocessor

### 2.5 Forward Pass Design
- Support both training and inference modes
- Efficient gate computation during inference
- Return dictionary with all necessary outputs for loss computation

## Deliverables
- [ ] `point_pillar_lite_bev_lidar_radar.py` with full model implementation
- [ ] Reliability computation module
- [ ] Spatial gate mechanism
- [ ] Unit tests for forward pass with different input shapes
- [ ] Model architecture documentation

## Success Criteria
- Model forward pass produces PSM and RM with correct shapes
- Gate values remain in [0, 1] range
- Compatible with standard detection losses
- No NaN or Inf values in outputs
- Inference runs efficiently without Teacher model
