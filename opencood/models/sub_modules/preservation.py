"""Frozen RadarDistill anchor and AP15 P-Heatmap preservation loss."""

import copy
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def checkpoint_sha256(path, chunk_size=1024 * 1024):
    """Return a streaming SHA-256 digest without modifying the checkpoint."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
    if not isinstance(checkpoint, dict):
        raise TypeError('Radar anchor checkpoint must contain a state dictionary.')
    return {
        (key[7:] if isinstance(key, str) and key.startswith('module.') else key): value
        for key, value in checkpoint.items()
    }


class FrozenRadarAnchor(nn.Module):
    """Radar-only copy of the selected E20 model used as a frozen supervisor."""

    _SOURCE_PREFIXES = {
        'vfe': 'radar_vfe.',
        'backbone': 'radar_backbone.',
        'distill': 'radar_distill.',
        'head': 'radar_head.',
    }
    _ALLOWED_MISSING_PREFIXES = ('distill.reliability_mask_head.',)

    def __init__(self, radar_vfe, radar_backbone, radar_distill, radar_head,
                 checkpoint_path, expected_sha256):
        super().__init__()
        self.vfe = copy.deepcopy(radar_vfe)
        self.backbone = copy.deepcopy(radar_backbone)
        self.distill = copy.deepcopy(radar_distill)
        self.head = copy.deepcopy(radar_head)
        self.checkpoint_path = checkpoint_path
        self.expected_sha256 = str(expected_sha256).lower()
        self.load_report = self._load_checkpoint()
        self.requires_grad_(False)
        self.eval()

    def _load_checkpoint(self):
        actual_sha256 = checkpoint_sha256(self.checkpoint_path)
        if self.expected_sha256 and actual_sha256 != self.expected_sha256:
            raise ValueError(
                'Radar anchor SHA-256 mismatch: expected %s, got %s for %s'
                % (self.expected_sha256, actual_sha256, self.checkpoint_path)
            )

        source = _unwrap_state_dict(torch.load(self.checkpoint_path, map_location='cpu'))
        target = self.state_dict()
        compatible = {}
        shape_mismatch = []
        for target_prefix, source_prefix in self._SOURCE_PREFIXES.items():
            destination_prefix = target_prefix + '.'
            for source_key, value in source.items():
                if not source_key.startswith(source_prefix):
                    continue
                destination_key = destination_prefix + source_key[len(source_prefix):]
                if destination_key in target and target[destination_key].shape == value.shape:
                    compatible[destination_key] = value
                elif destination_key in target:
                    shape_mismatch.append(destination_key)

        missing = sorted(set(target) - set(compatible))
        core_missing = [
            key for key in missing
            if not key.startswith(self._ALLOWED_MISSING_PREFIXES)
        ]
        if core_missing or shape_mismatch:
            raise RuntimeError(
                'Radar anchor checkpoint is not core-compatible; missing=%s, shape_mismatch=%s'
                % (core_missing[:20], shape_mismatch[:20])
            )

        target.update(compatible)
        self.load_state_dict(target, strict=True)
        return {
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_sha256': actual_sha256,
            'loaded_tensor_count': len(compatible),
            'allowed_missing_keys': missing,
            'core_missing_keys': core_missing,
            'shape_mismatch_keys': shape_mismatch,
        }

    def train(self, mode=True):
        # The anchor remains in inference mode even when its parent student is
        # switched to train mode. This also prevents BatchNorm buffer mutation.
        super().train(False)
        return self

    def forward(self, radar_points, gt_boxes):
        with torch.no_grad():
            batch_dict = {
                'batch_size': int(gt_boxes.shape[0]),
                'radar_points': radar_points,
                'gt_boxes': gt_boxes,
                'compute_loss': False,
            }
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone(batch_dict)
            batch_dict = self.distill(batch_dict)

            # Calling RadarCenterHead.forward() in eval mode performs decoding
            # and NMS. Preservation needs only the dense heatmap logits.
            feature = batch_dict[self.head.feature_key]
            shared_feature = self.head.shared_conv(feature)
            return [head(shared_feature) for head in self.head.heads_list]


class HeatmapPreservationLoss(nn.Module):
    """Normalized MSE/Smooth-L1 on Gaussian-weighted Car 3x3 responses."""

    def __init__(self, config, class_names_each_head):
        super().__init__()
        config = config or {}
        self.enabled = bool(config.get('enabled', False))
        self.loss_weight = float(config.get('lambda_pres', 0.0))
        self.target = config.get('target', 'P-Heatmap')
        self.region = config.get('region', 'car_gaussian_3x3')
        self.loss_type = str(config.get('loss_type', 'mse')).lower()
        self.normalization_scale = float(config.get('normalization_scale', 1.0))
        self.radius = int(config.get('region_radius', 1))
        self.sigma = float(config.get('gaussian_sigma', 1.0))

        if self.target != 'P-Heatmap':
            raise ValueError("AP15 preservation target must be 'P-Heatmap'.")
        if self.region != 'car_gaussian_3x3' or self.radius != 1:
            raise ValueError("AP15 preservation region must be 'car_gaussian_3x3' with radius 1.")
        if self.loss_type not in {'mse', 'smooth_l1'}:
            raise ValueError("preservation loss_type must be 'mse' or 'smooth_l1'.")
        if self.normalization_scale <= 0 or self.sigma <= 0:
            raise ValueError('preservation normalization_scale and gaussian_sigma must be positive.')
        if self.enabled and self.loss_weight <= 0:
            raise ValueError('enabled preservation requires lambda_pres > 0.')

        locations = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(locations, locations, indexing='ij')
        kernel = torch.exp(-0.5 * (xx.square() + yy.square()) / (self.sigma ** 2))
        self.register_buffer('gaussian_kernel', kernel.view(1, 1, 3, 3), persistent=True)

        self.car_head_index = None
        self.car_local_index = None
        for head_index, classes in enumerate(class_names_each_head):
            lowered = [str(value).lower() for value in classes]
            if 'car' in lowered:
                self.car_head_index = head_index
                self.car_local_index = lowered.index('car')
                break
        if self.car_head_index is None:
            raise ValueError("P-Heatmap preservation requires a CenterHead containing class 'car'.")
        if len(class_names_each_head[self.car_head_index]) != 1:
            raise ValueError('AP15 requires the selected Car head to contain only Car objects.')

    def _weighted_patch_map(self, probability):
        kernel = self.gaussian_kernel.to(device=probability.device, dtype=probability.dtype)
        numerator = F.conv2d(probability, kernel, padding=self.radius)
        normalizer = F.conv2d(torch.ones_like(probability), kernel, padding=self.radius)
        return numerator / normalizer.clamp_min(torch.finfo(probability.dtype).eps)

    def forward(self, student_pred_dicts, anchor_pred_dicts, target_dicts):
        head_index = self.car_head_index
        student_logits = student_pred_dicts[head_index]['hm'][:, self.car_local_index:self.car_local_index + 1]
        zero = student_logits.sum() * 0.0
        neutral = {
            'preservation_loss': 0.0,
            'preservation_weighted_loss': 0.0,
            'preservation_object_count': 0,
            'preservation_abs_standardized_drift': 0.0,
            'preservation_student_response_mean': 0.0,
            'preservation_anchor_response_mean': 0.0,
            'preservation_empty_object_case': 1,
        }
        if not self.enabled:
            return zero, neutral

        anchor_logits = anchor_pred_dicts[head_index]['hm'][:, self.car_local_index:self.car_local_index + 1]
        if student_logits.shape != anchor_logits.shape:
            raise ValueError(
                'Student/anchor Car heatmap shape mismatch: %s vs %s'
                % (tuple(student_logits.shape), tuple(anchor_logits.shape))
            )

        student_patch = self._weighted_patch_map(student_logits.sigmoid())[:, 0].flatten(1)
        anchor_patch = self._weighted_patch_map(anchor_logits.detach().sigmoid())[:, 0].flatten(1)
        object_indices = target_dicts['inds'][head_index].long()
        object_mask = target_dicts['masks'][head_index].bool()
        max_index = student_patch.shape[1] - 1
        safe_indices = object_indices.clamp(min=0, max=max_index)
        student_objects = student_patch.gather(1, safe_indices)[object_mask]
        anchor_objects = anchor_patch.gather(1, safe_indices)[object_mask]

        if student_objects.numel() == 0:
            return zero, neutral

        standardized_delta = (
            student_objects - anchor_objects
        ) / self.normalization_scale
        if self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(
                standardized_delta,
                torch.zeros_like(standardized_delta),
                reduction='mean',
            )
        else:
            loss = standardized_delta.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite AP15 preservation loss.')

        telemetry = {
            'preservation_loss': loss.detach().item(),
            'preservation_weighted_loss': (loss.detach() * self.loss_weight).item(),
            'preservation_object_count': int(student_objects.numel()),
            'preservation_abs_standardized_drift': standardized_delta.detach().abs().mean().item(),
            'preservation_student_response_mean': student_objects.detach().mean().item(),
            'preservation_anchor_response_mean': anchor_objects.detach().mean().item(),
            'preservation_empty_object_case': 0,
        }
        return loss, telemetry
