# Work Package 4: Multi-Stage Training Pipeline

## Objective
Implement and configure three-stage training process: Teacher → Distillation → Fusion.

## Main Goals
- Create training pipeline supporting distinct training stages
- Implement selective model freezing per stage
- Ensure proper Teacher model checkpointing and loading
- Provide clear logging and monitoring for each stage

## Key Tasks

### 4.1 Training Script Enhancement
- **File**: Extend `opencood/tools/train_w_kd.py` or create `opencood/tools/train_lite_bev.py`
- Support three distinct training stages via configuration
- Stage logic:
  - Stage 1 (teacher): Train LiDAR encoder, freeze nothing
  - Stage 2 (radar_distill): Load Teacher checkpoint, freeze it, train Radar branch
  - Stage 3 (fusion): Optionally freeze Teacher or fine-tune full model

### 4.2 YAML Configuration Templates
Create three configuration files:
- **`opencood/hypes_yaml/adver_city/adver_city_lite_bev_lidar_teacher.yaml`**
  - Stage: teacher
  - Model: point_pillar_lite_bev_lidar_radar
  - Loss: detection only
  
- **`opencood/hypes_yaml/adver_city/adver_city_lite_bev_radar_distill.yaml`**
  - Stage: radar_distill
  - Teacher checkpoint path
  - High distillation weight
  - Radar branch trainable, Teacher frozen
  
- **`opencood/hypes_yaml/adver_city/adver_city_lite_bev_fusion.yaml`**
  - Stage: fusion
  - All modules trainable
  - Optional low distillation weight
  - Full mixed-weather training

### 4.3 Model Freezing & Loading Logic
- Load Teacher checkpoint at start of Stage 2
- Implement freeze/unfreeze for specific submodules
- Support checkpoint resumption within stages
- Proper parameter group handling for optimizer

### 4.4 Comprehensive Logging
Track and log:
- Per-stage training progress
- Detection loss breakdown
- Distillation loss (when applicable)
- Mean gate value and gate collapse indicators
- Radar gate usage statistics
- Validation metrics after each epoch

### 4.5 Validation & Checkpointing
- Periodic validation between stages
- Save best checkpoints per stage
- Save final stage checkpoints for deployment
- Logging to TensorBoard/WandB for visualization

## Deliverables
- [ ] Training script supporting multi-stage pipeline
- [ ] Three YAML configuration files with proper stage definitions
- [ ] Model loading/freezing utilities
- [ ] Comprehensive logging and metrics tracking
- [ ] Stage transition documentation

## Success Criteria
- All three stages train without errors
- Teacher remains frozen in Stage 2
- Loss metrics logged correctly per stage
- Checkpoints save and load without issues
- Stage transitions are seamless with configuration changes
