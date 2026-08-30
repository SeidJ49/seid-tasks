"""Budget-preserving Car-only guidance for native RadarDistill PFD.

The module redistributes only the native PFD mass inside selected Car GT
Gaussians.  The rest map is copied exactly and the selected-class budget is
renormalized per batch item.  It is parameter-free and disabled by default.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from opencood.models.sub_modules.radar_evidence import RadarEvidence
from opencood.models.sub_modules.teacher_reliability import (
    TeacherReliability, _draw_gaussian, _gaussian_radius)


SUPPORTED_VARIANTS = {
    'native', 'teacher_only', 'radar_only', 'teacher_radar',
    'teacher_radar_doppler', 'car_weight_reference',
}


DEFAULT_CONFIG = {
    'enabled': False,
    'scope': {'mode': 'car_only'},
    'variant': 'teacher_radar',
    'teacher_reliability': {'mode': 'true_class_score', 'floor': 0.25},
    'radar_evidence': {
        'mode': 'point_count', 'point_kappa': 4.0, 'floor': 0.25,
        'use_doppler': False,
    },
    'combine': {'mode': 'geometric_mean', 'floor': 0.25},
    'budget': {
        'preserve_selected_class_object_budget': True,
        'preserve_rest_exactly': True,
    },
    'car_weight_reference_alpha': 1.0,
    'debug': False,
}


class GuidedPFD(nn.Module):
    """Redistribute native PFD within Car object support only."""

    def __init__(
            self, config, class_names, class_names_each_head,
            point_cloud_range, voxel_size, feature_map_stride=8,
            gaussian_overlap=0.1, min_radius=2, eps=1e-8):
        super(GuidedPFD, self).__init__()
        config = dict(config or {})
        if 'wp3_guided_pfd' in config:
            config = dict(config['wp3_guided_pfd'] or {})
        merged = dict(DEFAULT_CONFIG)
        merged.update(config)
        self.enabled = bool(merged['enabled'])
        self.variant = str(merged['variant'])
        self.scope_mode = str(merged.get('scope', {}).get('mode', 'car_only'))
        self.combine_mode = str(merged.get('combine', {}).get('mode', 'geometric_mean'))
        self.combine_floor = float(merged.get('combine', {}).get('floor', 0.25))
        self.preserve_selected_budget = bool(merged.get('budget', {}).get(
            'preserve_selected_class_object_budget', True))
        self.preserve_rest_exactly = bool(merged.get('budget', {}).get(
            'preserve_rest_exactly', True))
        self.reference_alpha = float(merged.get('car_weight_reference_alpha', 1.0))
        self.debug = bool(merged.get('debug', False))
        self.class_names = tuple(str(x) for x in class_names)
        self.class_names_each_head = tuple(tuple(str(x) for x in h) for h in class_names_each_head)
        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.voxel_size = tuple(float(x) for x in voxel_size)
        self.feature_map_stride = int(feature_map_stride)
        self.gaussian_overlap = float(gaussian_overlap)
        self.min_radius = int(min_radius)
        self.eps = float(eps)
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError('Unsupported guided PFD variant: {}'.format(self.variant))
        if self.scope_mode != 'car_only':
            raise ValueError('AP10 main implementation supports scope.mode=car_only only.')
        if self.combine_mode != 'geometric_mean':
            raise ValueError('AP10 main implementation requires geometric_mean.')
        if not 0.0 <= self.combine_floor <= 1.0:
            raise ValueError('combine.floor must be in [0, 1].')
        if not self.preserve_selected_budget or not self.preserve_rest_exactly:
            raise ValueError('AP10 requires both budget preservation invariants.')
        if 'car' not in self.class_names:
            raise ValueError('Car-only guided PFD requires class car.')
        self.car_class_index = self.class_names.index('car')

        teacher_cfg = dict(merged.get('teacher_reliability', {}))
        teacher_cfg.update({
            'enabled': True, 'detach_teacher': True,
            'apply_to_classes': ['car'], 'debug': self.debug,
        })
        radar_cfg = dict(merged.get('radar_evidence', {}))
        radar_cfg.update({
            'enabled': True, 'apply_to_classes': ['car'],
            'use_doppler': False, 'use_rcs': False, 'debug': self.debug,
        })
        doppler_cfg = dict(radar_cfg)
        doppler_cfg.update(dict(merged.get('radar_evidence', {})))
        doppler_cfg.update({
            'enabled': True, 'apply_to_classes': ['car'],
            'use_doppler': True, 'use_rcs': False, 'debug': self.debug,
        })
        geometry = dict(
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            feature_map_stride=self.feature_map_stride,
            gaussian_overlap=self.gaussian_overlap,
            min_radius=self.min_radius,
            eps=self.eps,
        )
        self.teacher = TeacherReliability(
            teacher_cfg, self.class_names, self.class_names_each_head, **geometry)
        self.radar = RadarEvidence(radar_cfg, self.class_names, **geometry)
        self.radar_doppler = RadarEvidence(doppler_cfg, self.class_names, **geometry)

    def _native_output(self, native_weight):
        batch, _, height, width = native_weight.shape
        empty_float = native_weight.new_empty((0,))
        empty_long = torch.empty((0,), device=native_weight.device, dtype=torch.long)
        return {
            'guided_weight': native_weight,
            'car_object_mask': native_weight.new_zeros((batch, 1, height, width)),
            'teacher_reliability_map': native_weight.new_ones((batch, 1, height, width)),
            'radar_evidence_map': native_weight.new_ones((batch, 1, height, width)),
            'combined_q_map': native_weight.new_ones((batch, 1, height, width)),
            'object_batch_index': empty_long, 'object_box_index': empty_long,
            'object_class_index': empty_long, 'teacher_score': empty_float,
            'teacher_margin': empty_float, 'teacher_reliability': empty_float,
            'radar_points': empty_float, 'radar_evidence': empty_float,
            'doppler_evidence': empty_float, 'combined_q': empty_float,
            'student_score': empty_float, 'teacher_student_gap': empty_float,
            'native_object_mass': empty_float, 'guided_object_mass': empty_float,
            'mass_ratio': empty_float,
            'native_total_mass': native_weight.sum((1, 2, 3)).detach(),
            'guided_total_mass': native_weight.sum((1, 2, 3)).detach(),
            'native_car_mass': native_weight.new_zeros((batch,)),
            'guided_car_mass': native_weight.new_zeros((batch,)),
            'native_rest_mass': native_weight.sum((1, 2, 3)).detach(),
            'guided_rest_mass': native_weight.sum((1, 2, 3)).detach(),
        }

    def _gaussians(self, gt_boxes, reference):
        batch, _, height, width = reference.shape
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_scale = self.voxel_size[0] * self.feature_map_stride
        y_scale = self.voxel_size[1] * self.feature_map_stride
        entries = []
        for batch_index in range(batch):
            for box_index in range(gt_boxes.shape[1]):
                box = gt_boxes[batch_index, box_index]
                class_id = int(round(float(box[-1].detach().item())))
                if class_id < 1 or class_id > len(self.class_names):
                    continue
                if float(box[3].detach().item()) <= 0 or float(box[4].detach().item()) <= 0:
                    continue
                cx = (float(box[0].detach().item()) - x_min) / x_scale
                cy = (float(box[1].detach().item()) - y_min) / y_scale
                ix, iy = int(cx), int(cy)
                if not (0 <= ix < width and 0 <= iy < height):
                    continue
                # CenterHead boxes are [x, y, z, dx, dy, dz, yaw, class].
                dx = float(box[3].detach().item()) / x_scale
                dy = float(box[4].detach().item()) / y_scale
                radius = max(int(_gaussian_radius(dx, dy, self.gaussian_overlap)), self.min_radius)
                gaussian = _draw_gaussian(height, width, ix, iy, radius, reference)
                entries.append((batch_index, box_index, class_id - 1, gaussian.detach()))
        return entries

    @staticmethod
    def _lookup(output, name):
        return {
            (int(b), int(i)): output[name][pos]
            for pos, (b, i) in enumerate(zip(
                output['object_batch_index'].tolist(), output['object_box_index'].tolist()))
        }

    def forward(
            self, native_weight, teacher_pred_dicts, radar_pred_dicts,
            radar_points, gt_boxes, variant_override=None):
        if native_weight.ndim != 4 or native_weight.shape[1] != 1:
            raise ValueError('native_weight must be [B,1,H,W].')
        variant = self.variant if variant_override is None else str(variant_override)
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError('Unsupported variant override: {}'.format(variant))
        if not self.enabled or variant == 'native':
            return self._native_output(native_weight)

        teacher = self.teacher(teacher_pred_dicts, gt_boxes, radar_pred_dicts)
        radar = self.radar(radar_points, gt_boxes, native_weight.shape[-2:])
        radar_doppler = self.radar_doppler(radar_points, gt_boxes, native_weight.shape[-2:])
        entries = self._gaussians(gt_boxes, native_weight)
        batch, _, height, width = native_weight.shape

        teacher_score = self._lookup(teacher, 'teacher_true_class_score')
        teacher_margin = self._lookup(teacher, 'teacher_margin')
        teacher_rel = self._lookup(teacher, 'reliability_with_floor')
        student_score = self._lookup(teacher, 'student_true_class_score')
        opportunity = self._lookup(teacher, 'distillation_opportunity')
        radar_count = self._lookup(radar, 'radar_point_count')
        radar_rel = self._lookup(radar, 'evidence_points_floor')
        doppler_rel = self._lookup(radar_doppler, 'evidence_combined_preview')

        car_sum = native_weight.new_zeros((batch, 1, height, width))
        noncar_support = torch.zeros((batch, 1, height, width), device=native_weight.device, dtype=torch.bool)
        q_numerator = native_weight.new_zeros((batch, 1, height, width))
        q_denominator = native_weight.new_zeros((batch, 1, height, width))
        object_values = []
        for batch_index, box_index, class_index, gaussian in entries:
            key = (batch_index, box_index)
            is_car = class_index == self.car_class_index
            if not is_car:
                noncar_support[batch_index, 0] |= gaussian > self.eps
                q = native_weight.new_tensor(1.0)
            else:
                t = teacher_rel[key].detach()
                r = radar_rel[key].detach()
                rd = doppler_rel[key].detach()
                if variant == 'teacher_only':
                    base = t
                elif variant == 'radar_only':
                    base = r
                elif variant == 'teacher_radar':
                    base = torch.sqrt(t * r)
                elif variant == 'teacher_radar_doppler':
                    base = torch.sqrt(t * rd)
                else:
                    base = native_weight.new_tensor(1.0)
                q = self.combine_floor + (1.0 - self.combine_floor) * base
                car_sum[batch_index, 0] += gaussian
                q_numerator[batch_index, 0] += gaussian * q
                q_denominator[batch_index, 0] += gaussian
            object_values.append((batch_index, box_index, class_index, gaussian, q, key))

        car_union = car_sum.clamp(0.0, 1.0)
        # Cross-class overlap belongs to rest so every non-Car Gaussian sees
        # the native map exactly. This enforces the explicit AP10 safety rule.
        car_mask = car_union * (~noncar_support).to(car_union.dtype)
        q_map = native_weight.new_ones((batch, 1, height, width))
        active = q_denominator > self.eps
        q_map[active] = (q_numerator / q_denominator.clamp_min(self.eps))[active]
        if variant == 'car_weight_reference':
            q_map = 1.0 + self.reference_alpha * car_union

        native_car = native_weight * car_mask
        native_rest = native_weight * (1.0 - car_mask)
        tentative = native_car * q_map.detach()
        guided_car = tentative.clone()
        for batch_index in range(batch):
            native_mass = native_car[batch_index].sum()
            tentative_mass = tentative[batch_index].sum()
            if native_mass > 0 and tentative_mass > self.eps:
                guided_car[batch_index] = tentative[batch_index] * (native_mass / tentative_mass)
            else:
                guided_car[batch_index] = native_car[batch_index]
        guided = guided_car + native_rest

        names = {
            'object_batch_index': [], 'object_box_index': [], 'object_class_index': [],
            'teacher_score': [], 'teacher_margin': [], 'teacher_reliability': [],
            'radar_points': [], 'radar_evidence': [], 'doppler_evidence': [],
            'combined_q': [], 'student_score': [], 'teacher_student_gap': [],
            'native_object_mass': [], 'guided_object_mass': [], 'mass_ratio': [],
        }
        for batch_index, box_index, class_index, gaussian, q, key in object_values:
            native_mass = (native_weight[batch_index, 0] * gaussian).sum()
            guided_mass = (guided[batch_index, 0] * gaussian).sum()
            names['object_batch_index'].append(batch_index)
            names['object_box_index'].append(box_index)
            names['object_class_index'].append(class_index)
            names['teacher_score'].append(teacher_score[key].detach())
            names['teacher_margin'].append(teacher_margin[key].detach())
            names['teacher_reliability'].append(teacher_rel[key].detach())
            names['radar_points'].append(radar_count[key].detach())
            names['radar_evidence'].append(radar_rel[key].detach())
            names['doppler_evidence'].append(doppler_rel[key].detach())
            names['combined_q'].append(q.detach())
            names['student_score'].append(student_score[key].detach())
            names['teacher_student_gap'].append(opportunity[key].detach())
            names['native_object_mass'].append(native_mass.detach())
            names['guided_object_mass'].append(guided_mass.detach())
            names['mass_ratio'].append((guided_mass / native_mass.clamp_min(self.eps)).detach())

        output = {
            'guided_weight': guided,
            'car_object_mask': car_mask.detach(),
            'teacher_reliability_map': teacher['reliability_map'].detach(),
            'radar_evidence_map': (radar_doppler if variant == 'teacher_radar_doppler' else radar)['evidence_map'].detach(),
            'combined_q_map': q_map.detach(),
            'native_total_mass': native_weight.sum((1, 2, 3)).detach(),
            'guided_total_mass': guided.sum((1, 2, 3)).detach(),
            'native_car_mass': native_car.sum((1, 2, 3)).detach(),
            'guided_car_mass': guided_car.sum((1, 2, 3)).detach(),
            'native_rest_mass': native_rest.sum((1, 2, 3)).detach(),
            # Rest is an explicit decomposition component.  Re-projecting the
            # final map with (1-mask) would double-attenuate soft mask pixels.
            'guided_rest_mass': native_rest.sum((1, 2, 3)).detach(),
            'native_rest_map': native_rest.detach(),
            'guided_rest_map': native_rest.detach(),
        }
        long_names = {'object_batch_index', 'object_box_index', 'object_class_index'}
        for name, values in names.items():
            if name in long_names:
                output[name] = torch.tensor(values, device=native_weight.device, dtype=torch.long)
            elif values:
                output[name] = torch.stack(values)
            else:
                output[name] = native_weight.new_empty((0,))
        return output
