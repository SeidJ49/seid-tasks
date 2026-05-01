# RadarDistill In OpenCOOD

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
