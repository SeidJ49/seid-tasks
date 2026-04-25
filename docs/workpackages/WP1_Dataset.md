# Work Package 1: Dataset Setup & Preparation

## Objective
Set up various datasets with different test scenarios for Lite-BEV LiDAR-Radar fusion model.

## Main Goals
- Create LiDAR-Radar intermediate fusion dataset for ADVERCity dataset
- Implement dataset with Doppler-based dynamic mask for motion-aware distillation
- Provide reliable data loading pipeline compatible with OpenCOOD framework

## Key Tasks

### 1.1 LiDAR-Radar Dataset Implementation
- **File**: `opencood/data_utils/datasets/adver_city/lidar_radar_intermediate_fusion_dataset_v1.py`
- **Goal**: Create new dataset class that provides:
  - `processed_lidar`: LiDAR point cloud data
  - `processed_radar`: Radar point cloud data
  - `label_dict`: Detection labels
  - `anchor_box`: Anchor boxes
  - `record_len`: Record length for temporal alignment
  - `pairwise_t_matrix`: Transformation matrices between agents
  - `dyn_mask_bev`: BEV-space dynamic mask for distillation

### 1.2 Doppler-based Dynamic Mask (M_dyn)
- Compute dynamic mask from Radar radial velocity (v_r)
- Apply hard threshold: |v_r| > 0.5 m/s
- Project to BEV coordinates
- Create per-pillar occupancy/max-aggregation
- Output: `dyn_mask_bev` in shape [B, 1, H, W]

### 1.3 Dataset Registry Extension
- **File**: `opencood/data_utils/datasets/__init__.py`
- Add new key: `LiDARRadarIntermediateFusionDatasetADVERCITY_V1`
- Ensure proper instantiation through dataset factory

### 1.4 Validation & Testing
- Verify dataset produces consistent tensor shapes
- Check that ego-contract keys are present
- Validate dyn_mask_bev is neither empty nor all-ones
- Test with training pipeline

## Deliverables
- [ ] `lidar_radar_intermediate_fusion_dataset_v1.py` with full implementation
- [ ] Updated dataset registry
- [ ] Test script validating dataset output structure
- [ ] Documentation of ego-contract requirements

## Success Criteria
- Dataset loads without errors
- All required keys present in output dictionary
- dyn_mask_bev contains meaningful dynamic information
- Compatible with existing OpenCOOD training pipeline
