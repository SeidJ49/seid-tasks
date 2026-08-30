"""Config-based teacher reliability for RadarDistill.

The module is a parameter-free sidecar.  It computes object-level reliability,
distillation opportunity, and a Gaussian-weighted spatial reliability map.  It
does not modify PFD or any other loss; AP10 may consume the returned map later.
"""

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn


SUPPORTED_MODES = {
    'true_class_score',
    'class_margin',
    'score_times_margin',
    'teacher_student_gap',
}

DEFAULT_CONFIG = {
    'enabled': False,
    'mode': 'true_class_score',
    'floor': 0.25,
    'temperature': 0.10,
    'detach_teacher': True,
    'apply_to_classes': ['car'],
    'debug': False,
}


def _gaussian_radius(height, width, min_overlap):
    """CenterNet radius formula using Python floats."""
    height = float(height)
    width = float(width)
    min_overlap = float(min_overlap)

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1 * b1 - 4.0 * a1 * c1, 0.0))) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2 * b2 - 4.0 * a2 * c2, 0.0))) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    if abs(a3) < 1e-12:
        r3 = min(r1, r2)
    else:
        r3 = (b3 + math.sqrt(max(b3 * b3 - 4.0 * a3 * c3, 0.0))) / 2.0
    return min(r1, r2, r3)


def _draw_gaussian(height, width, center_x, center_y, radius, reference):
    """Return one cropped 2-D CenterNet Gaussian on reference's device."""
    output = reference.new_zeros((height, width))
    radius = int(max(radius, 0))
    diameter = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(coords, coords)
    sigma = max(float(diameter) / 6.0, 1e-6)
    gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))

    center_x = int(center_x)
    center_y = int(center_y)
    left = min(center_x, radius)
    right = min(width - center_x, radius + 1)
    top = min(center_y, radius)
    bottom = min(height - center_y, radius + 1)
    if min(left + right, top + bottom) <= 0:
        return output
    output[center_y - top:center_y + bottom, center_x - left:center_x + right] = (
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    )
    return output


class TeacherReliability(nn.Module):
    """Compute detached teacher reliability at assigned GT centers.

    Args:
        config: Mapping containing ``wp3_teacher_reliability`` values or the
            values directly.
        class_names: Global class order used by GT class ids (1-based).
        class_names_each_head: Local class names for every CenterHead head.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: Model voxel size.
        feature_map_stride: CenterHead output stride.
        gaussian_overlap: CenterNet target Gaussian overlap.
        min_radius: Minimum target Gaussian radius.
        eps: Numerical activity threshold for the spatial average.
    """

    def __init__(
            self,
            config,
            class_names,
            class_names_each_head,
            point_cloud_range,
            voxel_size,
            feature_map_stride=8,
            gaussian_overlap=0.1,
            min_radius=2,
            eps=1e-8):
        super(TeacherReliability, self).__init__()
        config = dict(config or {})
        if 'wp3_teacher_reliability' in config:
            config = dict(config['wp3_teacher_reliability'] or {})
        merged = dict(DEFAULT_CONFIG)
        merged.update(config)

        self.enabled = bool(merged['enabled'])
        self.mode = str(merged['mode'])
        self.floor = float(merged['floor'])
        self.temperature = float(merged['temperature'])
        self.detach_teacher = bool(merged['detach_teacher'])
        self.apply_to_classes = tuple(str(name) for name in merged.get('apply_to_classes', []))
        self.debug = bool(merged['debug'])
        self.class_names = tuple(str(name) for name in class_names)
        self.class_names_each_head = tuple(tuple(str(name) for name in head) for head in class_names_each_head)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.voxel_size = tuple(float(value) for value in voxel_size)
        self.feature_map_stride = int(feature_map_stride)
        self.gaussian_overlap = float(gaussian_overlap)
        self.min_radius = int(min_radius)
        self.eps = float(eps)

        if self.mode not in SUPPORTED_MODES:
            raise ValueError("Unsupported teacher reliability mode '{}'".format(self.mode))
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError('Teacher reliability floor must be in [0, 1].')
        if self.temperature <= 0.0:
            raise ValueError('Teacher reliability temperature must be positive.')
        if not self.detach_teacher:
            raise ValueError('AP08 requires detach_teacher=true; teacher gradients are forbidden.')
        if self.feature_map_stride <= 0:
            raise ValueError('feature_map_stride must be positive.')
        if len(self.point_cloud_range) < 5 or len(self.voxel_size) < 2:
            raise ValueError('Invalid point-cloud geometry.')
        unknown = sorted(set(self.apply_to_classes).difference(self.class_names))
        if unknown:
            raise ValueError('Unknown apply_to_classes: {}'.format(unknown))

        flattened = [name for head in self.class_names_each_head for name in head]
        if sorted(flattened) != sorted(self.class_names) or len(flattened) != len(set(flattened)):
            raise ValueError('class_names_each_head must contain every global class exactly once.')
        self._class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self._selected_class_indices = set(
            self._class_to_index[name] for name in self.apply_to_classes
        )

    def extra_repr(self):
        return (
            'enabled={}, mode={}, floor={}, temperature={}, detach_teacher={}, '
            'apply_to_classes={}'
        ).format(
            self.enabled, self.mode, self.floor, self.temperature,
            self.detach_teacher, list(self.apply_to_classes)
        )

    def _global_score_maps(self, pred_dicts):
        if len(pred_dicts) != len(self.class_names_each_head):
            raise ValueError('Prediction-head count does not match class_names_each_head.')
        reference = pred_dicts[0]['hm']
        if reference.ndim != 4:
            raise ValueError('CenterHead heatmaps must have shape [B, C, H, W].')
        batch_size, _, height, width = reference.shape
        maps = reference.new_zeros((batch_size, len(self.class_names), height, width))
        for head_index, (pred, head_names) in enumerate(zip(pred_dicts, self.class_names_each_head)):
            logits = pred['hm']
            if logits.shape[0] != batch_size or tuple(logits.shape[-2:]) != (height, width):
                raise ValueError('All CenterHead heatmaps must share batch and spatial shape.')
            if logits.shape[1] != len(head_names):
                raise ValueError('Head {} has {} channels for {} classes.'.format(
                    head_index, logits.shape[1], len(head_names)
                ))
            scores = logits.sigmoid().detach()
            for local_index, name in enumerate(head_names):
                maps[:, self._class_to_index[name]] = scores[:, local_index]
        return maps.detach()

    @staticmethod
    def _empty_object_tensor(reference, dtype=None):
        return torch.empty((0,), device=reference.device, dtype=dtype or reference.dtype)

    def _identity_output(self, reference):
        batch_size, _, height, width = reference.shape
        empty_float = self._empty_object_tensor(reference)
        empty_long = self._empty_object_tensor(reference, dtype=torch.long)
        empty_bool = self._empty_object_tensor(reference, dtype=torch.bool)
        return {
            'reliability_map': reference.new_ones((batch_size, 1, height, width)).detach(),
            'object_batch_index': empty_long,
            'object_box_index': empty_long,
            'object_class_index': empty_long,
            'object_selected': empty_bool,
            'teacher_true_class_score': empty_float,
            'teacher_max_other_class_score': empty_float,
            'teacher_margin': empty_float,
            'student_true_class_score': empty_float,
            'distillation_opportunity': empty_float,
            'mode_signal_raw': empty_float,
            'reliability_raw': empty_float,
            'reliability_with_floor': empty_float,
        }

    def forward(self, teacher_pred_dicts, gt_boxes, student_pred_dicts=None):
        teacher_maps = self._global_score_maps(teacher_pred_dicts)
        if not self.enabled:
            return self._identity_output(teacher_maps)
        if gt_boxes.ndim != 3 or gt_boxes.shape[0] != teacher_maps.shape[0]:
            raise ValueError('gt_boxes must have shape [B, M, D] and match the prediction batch.')
        if gt_boxes.shape[-1] < 8:
            raise ValueError('gt_boxes must include seven box values and a 1-based class id.')

        student_maps = None
        if student_pred_dicts is not None:
            student_maps = self._global_score_maps(student_pred_dicts)
        if self.mode == 'teacher_student_gap' and student_maps is None:
            raise ValueError('teacher_student_gap mode requires student_pred_dicts.')

        batch_size, _, height, width = teacher_maps.shape
        numerator = teacher_maps.new_zeros((batch_size, 1, height, width))
        denominator = teacher_maps.new_zeros((batch_size, 1, height, width))
        values = {
            'object_batch_index': [],
            'object_box_index': [],
            'object_class_index': [],
            'object_selected': [],
            'teacher_true_class_score': [],
            'teacher_max_other_class_score': [],
            'teacher_margin': [],
            'student_true_class_score': [],
            'distillation_opportunity': [],
            'mode_signal_raw': [],
            'reliability_raw': [],
            'reliability_with_floor': [],
        }

        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_scale = self.voxel_size[0] * self.feature_map_stride
        y_scale = self.voxel_size[1] * self.feature_map_stride

        for batch_index in range(batch_size):
            for box_index in range(gt_boxes.shape[1]):
                box = gt_boxes[batch_index, box_index]
                class_id = int(round(float(box[-1].detach().item())))
                if class_id < 1 or class_id > len(self.class_names):
                    continue
                if float(box[3].detach().item()) <= 0.0 or float(box[4].detach().item()) <= 0.0:
                    continue
                center_x = (float(box[0].detach().item()) - x_min) / x_scale
                center_y = (float(box[1].detach().item()) - y_min) / y_scale
                center_x_int = int(center_x)
                center_y_int = int(center_y)
                if not (0 <= center_x_int < width and 0 <= center_y_int < height):
                    continue

                class_index = class_id - 1
                center_scores = teacher_maps[batch_index, :, center_y_int, center_x_int]
                true_score = center_scores[class_index]
                if len(self.class_names) > 1:
                    other_scores = torch.cat((center_scores[:class_index], center_scores[class_index + 1:]))
                    max_other = other_scores.max()
                else:
                    max_other = true_score.new_zeros(())
                margin = true_score - max_other
                margin_gate = torch.sigmoid(margin / self.temperature)

                if student_maps is None:
                    student_score = true_score.new_zeros(())
                    opportunity = true_score.new_zeros(())
                else:
                    student_score = student_maps[batch_index, class_index, center_y_int, center_x_int]
                    opportunity = torch.clamp(true_score - student_score, min=0.0)

                if self.mode == 'true_class_score':
                    mode_signal = true_score
                elif self.mode == 'class_margin':
                    mode_signal = margin_gate
                elif self.mode == 'score_times_margin':
                    mode_signal = true_score * margin_gate
                else:
                    mode_signal = opportunity
                mode_signal = mode_signal.clamp(0.0, 1.0)
                selected = class_index in self._selected_class_indices
                reliability_raw = mode_signal if selected else mode_signal.new_ones(())
                reliability_with_floor = (
                    self.floor + (1.0 - self.floor) * reliability_raw
                    if selected else reliability_raw.new_ones(())
                )

                values['object_batch_index'].append(batch_index)
                values['object_box_index'].append(box_index)
                values['object_class_index'].append(class_index)
                values['object_selected'].append(selected)
                values['teacher_true_class_score'].append(true_score)
                values['teacher_max_other_class_score'].append(max_other)
                values['teacher_margin'].append(margin)
                values['student_true_class_score'].append(student_score)
                values['distillation_opportunity'].append(opportunity)
                values['mode_signal_raw'].append(mode_signal)
                values['reliability_raw'].append(reliability_raw)
                values['reliability_with_floor'].append(reliability_with_floor)

                if selected:
                    dx = float(box[3].detach().item()) / x_scale
                    dy = float(box[4].detach().item()) / y_scale
                    radius = max(
                        int(_gaussian_radius(dx, dy, self.gaussian_overlap)),
                        self.min_radius,
                    )
                    gaussian = _draw_gaussian(
                        height, width, center_x_int, center_y_int, radius, teacher_maps
                    )
                    numerator[batch_index, 0] += gaussian * reliability_with_floor
                    denominator[batch_index, 0] += gaussian

        reliability_map = teacher_maps.new_ones((batch_size, 1, height, width))
        active = denominator > self.eps
        reliability_map[active] = (numerator / denominator.clamp_min(self.eps))[active]
        reliability_map = reliability_map.clamp(min=self.floor, max=1.0).detach()

        output = {'reliability_map': reliability_map}
        float_names = {
            'teacher_true_class_score', 'teacher_max_other_class_score', 'teacher_margin',
            'student_true_class_score', 'distillation_opportunity', 'mode_signal_raw',
            'reliability_raw', 'reliability_with_floor',
        }
        long_names = {'object_batch_index', 'object_box_index', 'object_class_index'}
        for name, entries in values.items():
            if not entries:
                if name in long_names:
                    output[name] = self._empty_object_tensor(teacher_maps, dtype=torch.long)
                elif name == 'object_selected':
                    output[name] = self._empty_object_tensor(teacher_maps, dtype=torch.bool)
                else:
                    output[name] = self._empty_object_tensor(teacher_maps)
            elif name in long_names:
                output[name] = torch.tensor(entries, device=teacher_maps.device, dtype=torch.long)
            elif name == 'object_selected':
                output[name] = torch.tensor(entries, device=teacher_maps.device, dtype=torch.bool)
            elif name in float_names:
                output[name] = torch.stack(entries).detach()
        return output
