# RadarDistill In OpenCOOD

Session update date: 06.05.2026

This note is the current runbook for the RadarDistill port in this OpenCOOD checkout.
It focuses on the training flow that is working in this repository now:

- train the LiDAR teacher first
- use that checkpoint for radar distillation
- optionally run joint training afterward

The active project root is:

```bash
/home/nj644/dev/Studis/seid-tasks/opencood
```

The Python environment used for the validated runs is:

```bash
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python
```

## What Is Implemented

- LiDAR teacher stage
- radar distillation stage
- joint stage
- multi-class TruckScenes setup with 5 classes:
	- `car`
	- `truck`
	- `bus`
	- `trailer`
	- `other_vehicle`
- teacher-to-radar initialization when `teacher_ckpt` is provided for distill or joint
- source-like RadarDistill block with the same CMA/AFD/PFD ideas implemented in `models/sub_modules/radardistill_block.py`

## Main Configs

- Teacher:
	- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_teacher.yaml`
- Distill:
	- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml`
- Joint:
	- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_joint.yaml`

## Training Stages

### 1. Teacher Training

Teacher training uses only the LiDAR teacher branch.

Relevant config values:

- `train_stage: teacher`
- `freeze_teacher: false`
- `input_source: ['lidar']`
- `label_type: 'lidar'`

Single-GPU command:

```bash
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/train.py \
	--hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_teacher.yaml
```

DDP example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python -m torch.distributed.launch \
	--nproc_per_node=2 --use_env \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/train_ddp.py \
	--hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_teacher.yaml
```

Example teacher run directory:

```bash
/home/nj644/dev/Studis/seid-tasks/opencood/logs/pillarnet_radar_teacher_2026_05_01_16_35_17
```

Example checkpoint from that run:

```bash
/home/nj644/dev/Studis/seid-tasks/opencood/logs/pillarnet_radar_teacher_2026_05_01_16_35_17/net_epoch18.pth
```

### 2. Distill Training

Distill training freezes the teacher branch, runs the radar branch, and optimizes:

- radar head loss
- RadarDistill feature loss

Relevant config values:

- `train_stage: distill`
- `freeze_teacher: true`
- `teacher_ckpt: '.../net_epoch18.pth'`

The current distill config already points to the latest validated teacher checkpoint example:

```bash
/home/nj644/dev/Studis/seid-tasks/opencood/logs/pillarnet_radar_teacher_2026_05_01_16_35_17/net_epoch18.pth
```

Single-GPU command:

```bash
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/train.py \
	--hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml
```

DDP example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python -m torch.distributed.launch \
	--nproc_per_node=2 --use_env \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/train_ddp.py \
	--hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml
```

Validated smoke-test behavior with the epoch-18 teacher checkpoint:

- distill forward succeeded on a real batch
- `radar_head_loss` was finite
- `distill_loss` was finite

### 3. Joint Training

Joint training keeps both the teacher and radar branches trainable and optimizes:

- teacher head loss
- radar head loss
- RadarDistill feature loss

Relevant config values:

- `train_stage: joint`
- `freeze_teacher: false`
- `teacher_ckpt: '.../net_epoch18.pth'`

Single-GPU command:

```bash
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/train.py \
	--hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_joint.yaml
```

DDP example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python -m torch.distributed.launch \
	--nproc_per_node=2 --use_env \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/train_ddp.py \
	--hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_joint.yaml
```

## Recommended Workflow

### Teacher First

1. Train the teacher config.
2. Pick the checkpoint you want to use, for example `net_epoch18.pth`.
3. Put that checkpoint path into:
	 - `pillarnet_radar_distill.yaml`
	 - `pillarnet_radar_joint.yaml`

### Then Distill

1. Run a one-batch smoke test if the model code changed.
2. Start full distill training.
3. Compare radar-only results against the original RadarDistill baseline.

### Joint Is Optional

Run joint only if you explicitly want to fine-tune teacher and student together after distillation.
For most reproduction work, teacher then distill is the more important path.

## Batch Size Notes

The YAMLs may still contain larger batch sizes, but memory use should be treated conservatively.

- Teacher training previously required reducing `batch_size` to `1` to avoid OOM on the tested GPU.
- Distill and joint can be heavier than teacher.
- If training crashes with CUDA OOM, reduce `train_params.batch_size` before retrying.

Safe starting point:

```yaml
train_params:
	batch_size: 1
```

## Resume Training

To continue from an OpenCOOD training directory, use `--model_dir`.

Example:

```bash
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/train.py \
	--hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml \
	--model_dir /home/nj644/dev/Studis/seid-tasks/opencood/logs/pillarnet_radar_distill_YYYY_MM_DD_HH_MM_SS
```

## Evaluation

Current OpenCOOD inference/eval commands use the integrated OpenCOOD AP evaluation path.

Single-run inference example:

```bash
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python \
	/home/nj644/dev/Studis/seid-tasks/opencood/tools/inference.py \

## Today’s Parity Fixes

The following gaps between RadarDistill/OpenPCDet and this OpenCOOD port were fixed during the current session.

### 1. Active TruckScenes Augmentation Path Is Now Wired

The active lidar+radar dataset path now actually applies the YAML-listed augmentations during training.

Relevant files:

- `data_utils/datasets/lidar_radar/single_dataset_lidar_radar_baseline.py`
- `data_utils/augmentor/data_augmentor.py`
- `data_utils/augmentor/augment_utils.py`

Implemented / enabled:

- `random_world_flip`
- `random_world_rotation`
- `random_world_scaling`
- `random_world_translation`

### 2. GT Sampling Is Integrated

OpenCOOD now uses a ground-truth database sampler for TruckScenes training, aligned with the RadarDistill training setup.

Relevant files:

- `data_utils/augmentor/database_sampler.py`
- `data_utils/augmentor/data_augmentor.py`

Active configs now include `gt_sampling`:

- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_teacher.yaml`
- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml`
- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_joint.yaml`

Important distinction:

- `train.pkl` / `val.pkl` / `test.pkl` are OpenCOOD scene PKLs used to enumerate scenes and sensors.
- `nuscenes_dbinfos_*sweeps_with_radar_withvelo.pkl` plus `gt_database_*` are the GT-sampling artifacts used to paste isolated objects into scenes.

Those are different files and both are needed for parity.

### 3. GT Boxes Are Filtered By LiDAR Support

TruckScenes GT boxes are now filtered by a minimum number of LiDAR hits before target generation, aligned with the RadarDistill teacher setup.

Relevant files:

- `utils/box_utils.py`
- `data_utils/post_processor/base_postprocessor.py`

Active configs now set:

```yaml
postprocess:
	filter_min_points_in_gt: 1
```

### 4. LiDAR Feature Count Matches RadarDistill Teacher Input

OpenCOOD now uses 5-feature LiDAR input for the active PillarNet teacher / distill / joint configs.

Current LiDAR feature layout in OpenCOOD:

- `x`
- `y`
- `z`
- `intensity`
- `timestamp`

Relevant file:

- `data_utils/datasets/lidar_radar/single_dataset_lidar_radar_baseline.py`

Current note about timestamp:

- The 5th channel is now present and propagated correctly.
- For the current 1-sweep OpenCOOD setup, the LiDAR timestamp value is zero.
- This matches feature-count parity, but not multi-sweep time-lag behavior.

### 5. Sweep Conclusion For The Current Teacher Reference

The current RadarDistill TruckScenes teacher reference is still a 1-sweep setup:

```yaml
MAX_SWEEPS: 1
```

Therefore:

- exact parity with that teacher does not require LiDAR multi-sweep loading
- the current OpenCOOD port is not missing a required LiDAR sweep feature for that reference run

## Current Explanation For The Metric Gap

The remaining gap was not explained by a single cause. The main issues identified were:

- augmentation listed in YAML but previously not applied in the active OpenCOOD dataset path
- missing `gt_sampling`
- missing GT min-points filtering
- LiDAR feature mismatch: 4 features vs 5 features

These affect all classes, not only `truck` and `trailer`, which is why even `car` differed.

## Fresh Training Requirement

All fixes above require a fresh training run.

They do not change the behavior of an already trained checkpoint.

## Optimization Differences

The original strong TruckScenes RadarDistill teacher does not use the old OpenCOOD optimizer setup.

### Old OpenCOOD Behavior

The earlier OpenCOOD teacher runs used:

- `AdamW`
- PyTorch `OneCycleLR`
- no source-matched fastai `OptimWrapper`
- no source-matched `adam_onecycle`
- no RadarDistill-style momentum scheduling

This older behavior was still able to produce a decent run, but it was not training with the same optimizer logic as RadarDistill.

### Current RadarDistill-Matched Behavior

The active parity teacher config now uses:

- `optimizer.core_method: adam_onecycle`
- RadarDistill-style fastai `OptimWrapper`
- RadarDistill-style `OneCycle`
- `betas: [0.9, 0.99]`
- `moms: [0.95, 0.85]`
- `div_factor: 10`
- `pct_start: 0.4`
- `grad_norm_clip: 10`

Relevant files:

- `tools/optimization_fastai.py`
- `tools/train_utils.py`
- `tools/train.py`
- `tools/train_ddp.py`

### Important Training-Loop Difference That Was Fixed

RadarDistill applies the per-batch one-cycle update before each training iteration.

OpenCOOD was initially stepping the scheduler after `optimizer.step()`. That means each batch was being trained with a shifted LR/momentum schedule.

This was fixed in:

- `tools/train.py`
- `tools/train_ddp.py`

### Why The Results Can Differ So Much

The large difference is not explained by one thing only.

The main reasons are:

- optimizer behavior changed from `AdamW + OneCycleLR` to RadarDistill-style `adam_onecycle`
- the scheduler timing was previously offset by one batch
- batch size is still different from the reference teacher
	- OpenCOOD teacher config currently uses `batch_size: 3`
	- RadarDistill reference uses `BATCH_SIZE_PER_GPU: 4`
- OpenCOOD resume behavior is not equivalent to RadarDistill checkpoint resume
	- OpenCOOD `--model_dir` reloads model weights only
	- it does not restore optimizer state and scheduler state like the original RadarDistill training loop
- GT sampling in OpenCOOD is still a simplified reimplementation, not the exact OpenPCDet sampler

Because of that, a run can show very different validation loss behavior even when final official TruckScenes metrics are already close.

### Current Interpretation

At this point the remaining gap looks only partly training-dependent.

What is probably training-dependent:

- some run-to-run variation
- some score calibration changes from batch size and optimizer changes

What is probably not just training noise:

- the class-specific `truck` gap
- the earlier drop in truck recall compared with RadarDistill
- the worse truck center/scale quality compared with RadarDistill

That means the optimizer port matters, but the remaining truck gap is still likely influenced by training-data parity, especially GT sampling quality.

### Retrain Rule

For the new optimizer comparison:

- start a fresh run from scratch
- do not resume an older `AdamW` run with `--model_dir`
- compare official TruckScenes metrics, not only validation loss

Current useful comparison points:

- older `AdamW` OpenCOOD run:
	- `/home/nj644/dev/Studis/seid-tasks/opencood/logs/pillarnet_radar_teacher_2026_05_07_09_31_20`
- newer `adam_onecycle` OpenCOOD run:
	- `/home/nj644/dev/Studis/seid-tasks/opencood/logs/pillarnet_radar_teacher_2026_05_07_11_29_19`
- RadarDistill reference:
	- `/home/nj644/dev/Studis/RadarDistill/output/truckscenes_models/pillarnet_vehicle/truckscenes_vehicle_v3_fixedgeom`

## Larger Dataset Workflow

If a larger TruckScenes training set is needed, two different artifact families must be rebuilt.

### A. OpenCOOD Scene PKLs

These are:

- `train.pkl`
- `val.pkl`
- `test.pkl`

They are generated by:

- `/home/nj644/dev/Studis/RadarDistill/pkl_generator_truckscences_opencood_sweeps_5_mini.py`

This script was updated to support CLI arguments instead of hardcoded mini-only paths.

It now supports:

- dataset root selection
- output directory selection
- split selection
- split ratios
- random seed
- number of stored radar sweeps

Example usage:

```bash
/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python \
	/home/nj644/dev/Studis/RadarDistill/pkl_generator_truckscences_opencood_sweeps_5_mini.py \
	--version v1.1-mini \
	--dataroot /path/to/man-truckscenes \
	--output-dir /path/to/output_trainval_dir \
	--split all \
	--train-ratio 0.6 \
	--val-ratio 0.2 \
	--test-ratio 0.2 \
	--seed 42 \
	--radar-sweeps 5
```

### B. OpenPCDet / RadarDistill Infos And GT Database

These are the artifacts used by the RadarDistill-style info loader and by `gt_sampling`:

- `nuscenes_infos_6radar_*sweeps_train.pkl`
- `nuscenes_infos_6radar_*sweeps_val.pkl`
- `nuscenes_dbinfos_*sweeps_with_radar_withvelo.pkl`
- `gt_database_*sweeps_with_radar_withvelo/`

They are generated by:

- `tools/create_truckscenes_data.py`

Example usage:

```bash
/home/nj644/dev/anaconda3/envs/opcdet/bin/python \
	/home/nj644/dev/Studis/RadarDistill/tools/create_truckscenes_data.py \
	--cfg_file /home/nj644/dev/Studis/RadarDistill/tools/cfgs/dataset_configs/truckscenes_dataset.yaml \
	--version v1.1-mini \
	--data_path /path/to/man-truckscenes \
	--save_path /path/to/man-truckscenes \
	--split_dir /path/to/splits \
	--create_db
```

### Important Rule For Bigger Data

If the training dataset gets bigger, regenerate both:

- OpenCOOD scene PKLs
- RadarDistill/OpenPCDet infos + dbinfos + gt_database

Rebuilding only one side is not enough.

## Notes On The DB File Name

The GT-sampling DB file currently used in OpenCOOD is named like:

- `nuscenes_dbinfos_1sweeps_with_radar_withvelo.pkl`

Even though the name starts with `nuscenes_`, in this project it is still the TruckScenes-generated database file stored under:

- `.../man-truckscenes/v1.1-mini/`

So the name is historical, but the contents are the correct TruckScenes DB infos.
	--model_dir /home/nj644/dev/Studis/seid-tasks/opencood/logs/pillarnet_radar_teacher_2026_05_01_16_35_17 \
	--fusion_method single \
	--eval_epoch 18
```

Official TruckScenes evaluation is still a separate pending integration step.

## Mapping To Original RadarDistill

- `teacher` stage corresponds to the LiDAR teacher training path.
- `distill` stage corresponds to the original RadarDistill training path where the teacher is frozen.
- `joint` is an OpenCOOD-added stage for training teacher and radar branches together.
- teacher-to-radar checkpoint cloning mirrors the idea of `ckpt.py`, but is done during model loading instead of by writing a separate rewritten checkpoint file.

## Practical Checklist

1. Build or confirm all required CUDA extensions and Python dependencies.
2. Train the teacher model.
3. Put the selected teacher checkpoint into the distill and joint YAMLs.
4. Run a one-batch distill smoke test if code changed.
5. Launch the full distill run.
6. Evaluate the resulting checkpoint.
7. Only then decide whether joint training is worth running.
# RadarDistill Integration In OpenCOOD

This note documents the RadarDistill port that was added to this OpenCOOD checkout, what is already working, and what is still not fully aligned with the original RadarDistill source code and paper.

## Summary

The current integration ports the main RadarDistill teacher-student architecture into OpenCOOD in a usable form.

What is implemented:

- LiDAR teacher branch with dynamic pillar VFE, sparse 2D pillar backbone, BEV backbone, and CenterHead-style detection head.
- Radar student branch with radar dynamic pillar VFE, sparse 2D pillar backbone, RadarDistill feature distillation block, and CenterHead-style radar head.
- Two-stage training flow inside one model: teacher stage and distillation stage.
- Teacher checkpoint loading for the teacher submodules.
- PillarNet-style configs for one-class training.
- Source-like CenterHead structure with `center`, `center_z`, `dim`, `rot`, `vel`, and `iou` heads.
- `AdamW` plus batch-stepped `OneCycleLR` training behavior.
- Dataset plumbing for raw LiDAR and radar points so the model can use dynamic pillars instead of only the old point-pillar voxel path.

What is not fully aligned yet:

- Full source multi-class and multi-head grouping is not restored yet. The current setup is intentionally one class first.
- Data augmentation is not yet ported to match the original RadarDistill pipeline.
- Direct loading of raw OpenPCDet RadarDistill checkpoints is not guaranteed. The current `teacher_ckpt` path is aligned primarily with the OpenCOOD-integrated teacher module names.
- The dataset and postprocess stack still includes some OpenCOOD legacy structure around voxel postprocessing, even though the integrated model uses its own CenterHead-style targets and decoding internally.

## Main Files Added Or Changed

### Models

- `models/point_pillar_radar_distill.py`
	- Main integrated RadarDistill model.
	- Contains the teacher path, radar path, stage switching, and teacher checkpoint loading.

- `models/pillarnet_radar_distill.py`
	- PillarNet-named alias for the same integrated model.
	- Use this for the new paper-style configs.

### Model Submodules

- `models/sub_modules/radardistill_vfe.py`
	- Dynamic pillar VFE for LiDAR and radar.
	- Uses `torch_scatter` when available and keeps a fallback path.

- `models/sub_modules/radardistill_spconv_backbone.py`
	- Sparse 2D pillar backbones for teacher and radar branches.

- `models/sub_modules/radardistill_bev_backbone.py`
	- BEV backbone matching the source structure more closely.

- `models/sub_modules/radardistill_block.py`
	- Feature distillation module and distillation losses.

- `models/sub_modules/radardistill_head.py`
	- Shared CenterHead-style detection head used for teacher and radar branches.
	- Includes `vel` and `iou` outputs in the current one-class aligned setup.

### Dataset And Training Plumbing

- `data_utils/datasets/lidar_radar/single_dataset_lidar_radar_baseline.py`
	- Extended so raw `lidar_points` and `radar_points` are preserved and batched.
	- Also updated to work with model-native `final_box_dict` outputs.

- `loss/radardistill_loss.py`
	- Wrapper loss used by the OpenCOOD trainer.

- `tools/train_utils.py`
	- Updated to support `onecycle` scheduler creation.

- `tools/train.py`
	- Updated so schedulers can step per batch, which is required for the source-like one-cycle setup.

- `hypes_yaml/yaml_utils.py`
	- Added `load_pillarnet_radar_params` so model voxel size and preprocessing voxel size can differ.

### Configs

- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_teacher.yaml`
	- Teacher-stage training config.
	- 20 epochs.

- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml`
	- Distillation-stage training config.
	- 40 epochs.

## Fully Aligned

The following parts are reasonably aligned with the RadarDistill architecture and training recipe:

- Teacher-student structure exists in OpenCOOD.
- Separate teacher and distillation stages exist.
- PillarNet-style body is used instead of the old PointPillar scatter-style body.
- Source-like head structure is restored for the one-class case.
- One-cycle training support is implemented.
- The paper-style stage schedule is available: teacher for 20 epochs and distill for 40 epochs.

## Partially Aligned

The following parts are only partially aligned:

- Head structure is source-like, but currently only configured for one class.
- Checkpoint loading works for the integrated OpenCOOD teacher path, but not as a guaranteed raw source-checkpoint conversion path.
- Range and voxel setup are moved closer to the source architecture, but are still adapted to the current OpenCOOD dataset setup.

## Not Yet Aligned

The following parts are still missing or simplified compared with the original RadarDistill source:

- Full multi-class, multi-head source configuration.
- Source-equivalent augmentation pipeline.
- End-to-end confirmation on full real training and evaluation runs.
- Direct source checkpoint conversion for arbitrary original RadarDistill weights.

## Why The New Config Uses PillarNet

The original integration file was named `point_pillar_radar_distill.py`, but the architecture being integrated is structurally closer to PillarNet than to the classic PointPillar stack.

For that reason, a new alias and config path were added:

- `models/pillarnet_radar_distill.py`
- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_teacher.yaml`
- `hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml`

These are the recommended entry points for this integration.

## Dependencies

RadarDistill relies on the copied deformable convolution package and scatter ops.

Build deformable convolution kernels:

```bash
cd /home/nj644/dev/Studis/seid-tasks/opencood/pcdet_utils/basicblock
python setup.py build_ext --inplace
```

Install `torch_scatter` inside the active conda environment:

```bash
pip install torch_scatter
```

## Training

### 1. Train The Teacher

```bash
python /home/nj644/dev/Studis/seid-tasks/opencood/tools/train.py \
  --hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_teacher.yaml
```

This trains the LiDAR teacher branch for 20 epochs.

### 2. Train Distillation

First set `teacher_ckpt` inside `pillarnet_radar_distill.yaml` to the teacher checkpoint you want to use.

Then run:

```bash
python /home/nj644/dev/Studis/seid-tasks/opencood/tools/train.py \
  --hypes_yaml /home/nj644/dev/Studis/seid-tasks/opencood/hypes_yaml/truckscences_lidar_radar/pillarnet_radar_distill.yaml
```

This trains the radar student with the teacher frozen.

## Validation Status

What has already been validated:

- New PillarNet teacher config constructs successfully.
- New PillarNet distillation config constructs successfully.
- `AdamW` plus `OneCycleLR` setup works.
- Synthetic teacher-stage forward pass runs.
- Synthetic distillation-stage forward pass runs.
- The one-class source-style head with `vel` and `iou` now produces finite training losses in the synthetic smoke test.

What still needs full validation:

- Real dataloader batch smoke test on the TruckScenes/OpenCOOD dataset.
- Full teacher training run.
- Full distillation training run.
- Evaluation and box-quality comparison against the original RadarDistill implementation.

## Recommendation

Use the current integration as a strong first one-class PillarNet port of RadarDistill into OpenCOOD, not as a claim of full one-to-one reproduction yet.

The implementation is structurally close enough to start real training, but it is still a partial alignment rather than a finished reproduction of every source detail.
