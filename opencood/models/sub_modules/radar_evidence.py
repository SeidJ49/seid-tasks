"""Object-wise radar evidence for RadarDistill.

The parameter-free sidecar counts real radar returns inside oriented BEV GT
boxes and maps them through a saturating function.  It does not modify PFD or
any training loss.  Doppler is optional; RCS is exposed as a debug statistic
only and never acts as a hard gate.
"""

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn


DEFAULT_CONFIG = {
    'enabled': False,
    'mode': 'point_count',
    'point_kappa': 4.0,
    'floor': 0.25,
    'apply_to_classes': ['car'],
    'use_doppler': False,
    'use_rcs': False,
    'doppler_mapping': 'sigmoid',
    'doppler_mu': 1.0,
    'doppler_tau': 0.75,
    'doppler_floor': 0.25,
    'combination': 'geometric_mean',
    'doppler_modifier_floor': 0.50,
    'batch_index': 0,
    'x_index': 1,
    'y_index': 2,
    'doppler_index': 4,
    'rcs_index': 5,
    'debug': False,
}


def point_count_evidence(counts, kappa):
    """Saturating point-count evidence 1 - exp(-N/kappa)."""
    if float(kappa) <= 0.0:
        raise ValueError('point_kappa must be positive.')
    counts = torch.clamp(counts, min=0.0)
    return 1.0 - torch.exp(-counts / float(kappa))


def _gaussian_radius(height, width, min_overlap):
    height = float(height)
    width = float(width)
    min_overlap = float(min_overlap)
    a1, b1 = 1.0, height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1 * b1 - 4.0 * a1 * c1, 0.0))) / 2.0
    a2, b2 = 4.0, 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2 * b2 - 4.0 * a2 * c2, 0.0))) / 2.0
    a3, b3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    r3 = r2 if abs(a3) < 1e-12 else (
        b3 + math.sqrt(max(b3 * b3 - 4.0 * a3 * c3, 0.0))) / (2.0 * a3)
    return min(r1, r2, r3)


def _draw_gaussian(height, width, center_x, center_y, radius, reference):
    output = reference.new_zeros((height, width))
    radius = int(max(radius, 0))
    coords = torch.arange(-radius, radius + 1, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(coords, coords)
    sigma = max(float(2 * radius + 1) / 6.0, 1e-6)
    gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    center_x, center_y = int(center_x), int(center_y)
    left, right = min(center_x, radius), min(width - center_x, radius + 1)
    top, bottom = min(center_y, radius), min(height - center_y, radius + 1)
    if min(left + right, top + bottom) <= 0:
        return output
    output[center_y - top:center_y + bottom, center_x - left:center_x + right] = (
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    )
    return output


class RadarEvidence(nn.Module):
    """Compute object-level evidence and a Gaussian-weighted spatial map.

    ``radar_points`` uses the collated BM2CP layout
    ``[batch_idx, x, y, z, radial_velocity, rcs]`` by default.  ``gt_boxes``
    uses ``[x, y, z, h, w, l, yaw, class_id]`` with 1-based class ids.
    Physical point assignment uses l/w (indices 5/4), validated by AP10-B.
    The diagnostic Gaussian retains the historical 3/4 target convention
    under AP10-C's KEEP_NATIVE_FOR_COMPARABILITY policy.
    """

    def __init__(
            self, config, class_names, point_cloud_range, voxel_size,
            feature_map_stride=8, gaussian_overlap=0.1, min_radius=2,
            eps=1e-8):
        super(RadarEvidence, self).__init__()
        config = dict(config or {})
        if 'wp3_radar_evidence' in config:
            config = dict(config['wp3_radar_evidence'] or {})
        merged = dict(DEFAULT_CONFIG)
        merged.update(config)
        self.enabled = bool(merged['enabled'])
        self.mode = str(merged['mode'])
        self.point_kappa = float(merged['point_kappa'])
        self.floor = float(merged['floor'])
        self.apply_to_classes = tuple(str(x) for x in merged['apply_to_classes'])
        self.use_doppler = bool(merged['use_doppler'])
        self.use_rcs = bool(merged['use_rcs'])
        self.doppler_mapping = str(merged['doppler_mapping'])
        self.doppler_mu = float(merged['doppler_mu'])
        self.doppler_tau = float(merged['doppler_tau'])
        self.doppler_floor = float(merged['doppler_floor'])
        self.combination = str(merged['combination'])
        self.doppler_modifier_floor = float(merged['doppler_modifier_floor'])
        self.batch_index = int(merged['batch_index'])
        self.x_index = int(merged['x_index'])
        self.y_index = int(merged['y_index'])
        self.doppler_index = int(merged['doppler_index'])
        self.rcs_index = int(merged['rcs_index'])
        self.debug = bool(merged['debug'])
        self.class_names = tuple(str(x) for x in class_names)
        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.voxel_size = tuple(float(x) for x in voxel_size)
        self.feature_map_stride = int(feature_map_stride)
        self.gaussian_overlap = float(gaussian_overlap)
        self.min_radius = int(min_radius)
        self.eps = float(eps)
        if self.mode != 'point_count':
            raise ValueError('AP09 supports point_count as the primary mode.')
        if self.point_kappa <= 0.0:
            raise ValueError('point_kappa must be positive.')
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError('floor must be in [0, 1].')
        if not 0.0 <= self.doppler_floor <= 1.0:
            raise ValueError('doppler_floor must be in [0, 1].')
        if not 0.0 <= self.doppler_modifier_floor <= 1.0:
            raise ValueError('doppler_modifier_floor must be in [0, 1].')
        if self.doppler_tau <= 0.0:
            raise ValueError('doppler_tau must be positive.')
        if self.doppler_mapping != 'sigmoid':
            raise ValueError('The online sidecar uses sigmoid Doppler mapping; quantile_rank is offline only.')
        if self.combination not in {'geometric_mean', 'multiplicative_modifier'}:
            raise ValueError('Unknown point/Doppler combination.')
        unknown = sorted(set(self.apply_to_classes).difference(self.class_names))
        if unknown:
            raise ValueError('Unknown apply_to_classes: {}'.format(unknown))
        self._class_to_index = {name: i for i, name in enumerate(self.class_names)}
        self._selected = set(self._class_to_index[name] for name in self.apply_to_classes)

    @staticmethod
    def _empty(reference, dtype=None):
        return torch.empty((0,), device=reference.device, dtype=dtype or reference.dtype)

    def _identity(self, reference, batch_size, height, width):
        empty_float = self._empty(reference)
        empty_long = self._empty(reference, torch.long)
        empty_bool = self._empty(reference, torch.bool)
        return {
            'evidence_map': reference.new_ones((batch_size, 1, height, width)),
            'object_batch_index': empty_long,
            'object_box_index': empty_long,
            'object_class_index': empty_long,
            'object_selected': empty_bool,
            'radar_point_count': empty_float,
            'abs_doppler_mean': empty_float,
            'doppler_available': empty_bool,
            'rcs_mean': empty_float,
            'rcs_available': empty_bool,
            'evidence_points_raw': empty_float,
            'evidence_points_floor': empty_float,
            'evidence_doppler': empty_float,
            'evidence_combined_preview': empty_float,
        }

    def _inside_oriented_bev(self, points, box):
        dx = points[:, self.x_index] - box[0]
        dy = points[:, self.y_index] - box[1]
        cosine, sine = torch.cos(box[6]), torch.sin(box[6])
        local_x = dx * cosine + dy * sine
        local_y = -dx * sine + dy * cosine
        # AP10-A: runtime GT is hwl; longitudinal BEV extent is length.
        return (local_x.abs() <= box[5] / 2.0) & (local_y.abs() <= box[4] / 2.0)

    def forward(self, radar_points, gt_boxes, feature_map_shape, gt_mask=None):
        if gt_boxes.ndim != 3 or gt_boxes.shape[-1] < 8:
            raise ValueError('gt_boxes must be [B, M, >=8].')
        if radar_points.ndim != 2:
            raise ValueError('radar_points must be [N, F].')
        height, width = int(feature_map_shape[0]), int(feature_map_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError('feature_map_shape must be positive.')
        batch_size = int(gt_boxes.shape[0])
        reference = radar_points if radar_points.numel() else gt_boxes
        if not self.enabled:
            return self._identity(reference, batch_size, height, width)
        if gt_mask is not None and tuple(gt_mask.shape) != tuple(gt_boxes.shape[:2]):
            raise ValueError('gt_mask must match gt_boxes[:2].')
        required = max(self.batch_index, self.x_index, self.y_index)
        if radar_points.shape[1] <= required:
            raise ValueError('radar_points do not contain batch/x/y columns.')

        numerator = reference.new_zeros((batch_size, 1, height, width))
        denominator = reference.new_zeros((batch_size, 1, height, width))
        fields = {name: [] for name in [
            'object_batch_index', 'object_box_index', 'object_class_index',
            'object_selected', 'radar_point_count', 'abs_doppler_mean',
            'doppler_available', 'rcs_mean', 'rcs_available',
            'evidence_points_raw', 'evidence_points_floor',
            'evidence_doppler', 'evidence_combined_preview',
        ]}
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_scale = self.voxel_size[0] * self.feature_map_stride
        y_scale = self.voxel_size[1] * self.feature_map_stride

        for batch_index in range(batch_size):
            batch_points = radar_points[
                radar_points[:, self.batch_index].long() == batch_index
            ] if radar_points.numel() else radar_points
            for box_index in range(gt_boxes.shape[1]):
                if gt_mask is not None and not bool(gt_mask[batch_index, box_index].item()):
                    continue
                box = gt_boxes[batch_index, box_index]
                class_id = int(round(float(box[-1].detach().item())))
                if class_id < 1 or class_id > len(self.class_names):
                    continue
                if any(float(box[index].detach().item()) <= 0.0 for index in (3, 4, 5)):
                    continue
                center_x = (float(box[0].detach().item()) - x_min) / x_scale
                center_y = (float(box[1].detach().item()) - y_min) / y_scale
                center_x_int, center_y_int = int(center_x), int(center_y)
                if not (0 <= center_x_int < width and 0 <= center_y_int < height):
                    continue
                inside = self._inside_oriented_bev(batch_points, box) if batch_points.numel() else torch.zeros(
                    (0,), device=reference.device, dtype=torch.bool)
                object_points = batch_points[inside]
                count = reference.new_tensor(float(object_points.shape[0]))
                points_raw = point_count_evidence(count, self.point_kappa)
                selected = (class_id - 1) in self._selected
                points_floor = self.floor + (1.0 - self.floor) * points_raw if selected else points_raw.new_ones(())

                doppler_available = bool(object_points.shape[0] > 0 and object_points.shape[1] > self.doppler_index)
                if doppler_available:
                    doppler_values = object_points[:, self.doppler_index].abs()
                    doppler_values = doppler_values[torch.isfinite(doppler_values)]
                    doppler_available = bool(doppler_values.numel() > 0)
                if doppler_available:
                    doppler_mean = doppler_values.mean()
                    doppler_raw = torch.sigmoid((doppler_mean - self.doppler_mu) / self.doppler_tau)
                else:
                    doppler_mean = reference.new_zeros(())
                    doppler_raw = reference.new_zeros(())
                doppler_effective = self.doppler_floor + (1.0 - self.doppler_floor) * doppler_raw

                rcs_available = bool(object_points.shape[0] > 0 and object_points.shape[1] > self.rcs_index)
                if rcs_available:
                    rcs_values = object_points[:, self.rcs_index]
                    rcs_values = rcs_values[torch.isfinite(rcs_values)]
                    rcs_available = bool(rcs_values.numel() > 0)
                rcs_mean = rcs_values.mean() if rcs_available else reference.new_zeros(())

                if not selected:
                    combined = points_raw.new_ones(())
                elif not self.use_doppler:
                    combined = points_floor
                elif self.combination == 'geometric_mean':
                    combined = torch.sqrt(points_floor * doppler_effective)
                else:
                    modifier = self.doppler_modifier_floor + (1.0 - self.doppler_modifier_floor) * doppler_raw
                    combined = points_floor * modifier
                combined = combined.clamp(0.0, 1.0)

                values = {
                    'object_batch_index': batch_index, 'object_box_index': box_index,
                    'object_class_index': class_id - 1, 'object_selected': selected,
                    'radar_point_count': count, 'abs_doppler_mean': doppler_mean,
                    'doppler_available': doppler_available, 'rcs_mean': rcs_mean,
                    'rcs_available': rcs_available, 'evidence_points_raw': points_raw,
                    'evidence_points_floor': points_floor, 'evidence_doppler': doppler_raw,
                    'evidence_combined_preview': combined,
                }
                for name, value in values.items():
                    fields[name].append(value)

                if selected:
                    dx = float(box[3].detach().item()) / x_scale
                    dy = float(box[4].detach().item()) / y_scale
                    radius = max(int(_gaussian_radius(dx, dy, self.gaussian_overlap)), self.min_radius)
                    gaussian = _draw_gaussian(height, width, center_x_int, center_y_int, radius, reference)
                    numerator[batch_index, 0] += gaussian * combined
                    denominator[batch_index, 0] += gaussian

        evidence_map = reference.new_ones((batch_size, 1, height, width))
        active = denominator > self.eps
        evidence_map[active] = (numerator / denominator.clamp_min(self.eps))[active]
        output = {'evidence_map': evidence_map.clamp(0.0, 1.0).detach()}
        long_names = {'object_batch_index', 'object_box_index', 'object_class_index'}
        bool_names = {'object_selected', 'doppler_available', 'rcs_available'}
        for name, entries in fields.items():
            if not entries:
                output[name] = self._empty(reference, torch.long if name in long_names else torch.bool if name in bool_names else None)
            elif name in long_names:
                output[name] = torch.tensor(entries, device=reference.device, dtype=torch.long)
            elif name in bool_names:
                output[name] = torch.tensor(entries, device=reference.device, dtype=torch.bool)
            else:
                output[name] = torch.stack(entries).detach()
        return output
