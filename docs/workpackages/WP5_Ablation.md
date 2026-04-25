# Work Package 5: Baseline & Ablation Experiments

## Objective
Set up baseline models and ablation variants to validate Lite-BEV design decisions.

## Main Goals
- Establish performance baselines for comparison
- Validate individual components (gate, reliability, distillation)
- Create reproducible experiment configurations
- Prepare systematic evaluation framework

## Key Tasks

### 5.1 Baseline Variants
Implement and configure following baselines via YAML flags:

1. **LiDAR Only**
   - Use only LiDAR encoder and detection head
   - Flag: `use_gate=false`, `use_radar=false`

2. **Radar Only**
   - Use only Radar encoder and detection head
   - Flag: `use_gate=false`, `use_lidar=false`

3. **Naive Fusion (Concat/Sum)**
   - Simple concatenation or sum of LiDAR and Radar features
   - No gate mechanism
   - Flag: `use_gate=false`, `fusion_type=concat`

4. **Distillation Only**
   - Full Lite-BEV with distillation, no spatial gate
   - Flag: `use_gate=false`, `use_distillation=true`

5. **Gate Without Reliability**
   - Spatial gate trained from [F_L, F_R] only
   - No reliability map U_L
   - Flag: `use_reliability=false`, `use_gate=true`

6. **Full Lite-BEV** (Complete implementation)
   - All components: gate, reliability, distillation
   - Flag: `use_gate=true`, `use_reliability=true`, `use_distillation=true`

### 5.2 YAML Configuration Variants
- Create separate YAML files for each variant in `opencood/hypes_yaml/adver_city/`
- Name pattern: `adver_city_lite_bev_{variant_name}.yaml`
- Document YAML differences for reproducibility

### 5.3 Ablation Study Framework
- Systematic script to run all variants with identical training setup
- Same random seed, learning rate, batch size across all variants
- Identical validation schedule
- Consistent checkpoint naming for comparison

### 5.4 Evaluation Metrics
- Standard detection metrics: AP, AP_3D, mAP
- Per-sensor contribution analysis
- Gate collapse detection (mean gate = 0 or 1)
- Inference speed and model size comparison

### 5.5 Results Documentation
- Ablation results table showing all variants
- Visualization of gate maps across scenarios
- Performance vs. model complexity analysis
- Lessons learned from each component

## Deliverables
- [ ] YAML files for all 6 baseline variants
- [ ] Ablation study runner script
- [ ] Baseline configuration documentation
- [ ] Results summary and comparison tables
- [ ] Component contribution analysis

## Success Criteria
- All baselines train successfully
- Clear performance ranking: naive < baselines < full Lite-BEV
- Gate component validated as beneficial
- Reliability component improves robustness
- Distillation improves Radar performance
