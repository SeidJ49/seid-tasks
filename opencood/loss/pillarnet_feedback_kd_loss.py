import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarnetFeedbackKdLoss(nn.Module):
    """PillarNet detection loss plus configurable feedback KD losses.

    The original scale-normalized BEV feature KD remains the default behavior.
    Optional flags add the feedback ablations: LiDAR-teacher foreground hybrid
    masks, instance-level CenterHead distillation, and inverse object-area
    rebalancing for small/large object imbalance.
    """

    def __init__(self, args):
        super().__init__()
        self.kd = args.get('kd', {})
        self.loss_dict = {}

    @staticmethod
    def _channel_rms(feature, eps):
        return feature.pow(2).mean(dim=(-2, -1), keepdim=True).add(eps).sqrt()

    def _normalize_feature(self, feature):
        norm_type = self.kd.get('feature_normalization', 'channel_rms').lower()
        eps = float(self.kd.get('normalization_eps', 1e-6))

        if norm_type in {'none', 'identity', 'off'}:
            return feature
        if norm_type == 'channel_rms':
            return feature / self._channel_rms(feature, eps)
        if norm_type == 'channel_l2':
            return F.normalize(feature, p=2, dim=1, eps=eps)
        if norm_type == 'channel_standardize':
            mean = feature.mean(dim=(-2, -1), keepdim=True)
            std = feature.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
            return (feature - mean) / std

        raise ValueError(
            f"Unsupported KD feature_normalization='{norm_type}'. "
            "Use channel_rms, channel_l2, channel_standardize, or none."
        )

    @staticmethod
    def _resize_mask(mask, size, mode='bilinear'):
        if mask.shape[-2:] == size:
            return mask
        if mode == 'nearest':
            return F.interpolate(mask, size=size, mode=mode)
        return F.interpolate(mask, size=size, mode=mode, align_corners=False)

    @staticmethod
    def _safe_mean(value):
        if value is None:
            return 0.0
        return value.detach().mean().item()

    def _heatmap_to_foreground(self, pred_dicts, target_hw=None):
        if not pred_dicts:
            return None

        masks = []
        threshold = float(self.kd.get('teacher_foreground_threshold', 0.10))
        normalize = bool(self.kd.get('teacher_foreground_normalize', False))
        eps = float(self.kd.get('normalization_eps', 1e-6))
        for pred_dict in pred_dicts:
            if 'hm' not in pred_dict:
                continue
            hm = torch.sigmoid(pred_dict['hm'].detach())
            hm = hm.max(dim=1, keepdim=True)[0]
            hm = ((hm - threshold) / max(1.0 - threshold, eps)).clamp(0.0, 1.0)
            if target_hw is not None and hm.shape[-2:] != target_hw:
                hm = self._resize_mask(hm, target_hw)
            masks.append(hm)

        if not masks:
            return None
        foreground = torch.stack(masks, dim=0).max(dim=0)[0]
        if normalize:
            flat = foreground.flatten(1)
            max_val = flat.max(dim=1)[0].view(-1, 1, 1, 1).clamp_min(eps)
            foreground = foreground / max_val
        return foreground.clamp(0.0, 1.0)

    def _feature_to_foreground(self, feature):
        if feature is None:
            return None
        eps = float(self.kd.get('normalization_eps', 1e-6))
        activation = feature.detach().abs().mean(dim=1, keepdim=True)
        flat = activation.flatten(1)
        min_val = flat.min(dim=1)[0].view(-1, 1, 1, 1)
        max_val = flat.max(dim=1)[0].view(-1, 1, 1, 1)
        return ((activation - min_val) / (max_val - min_val).clamp_min(eps)).clamp(0.0, 1.0)

    def _teacher_foreground_mask(self, output_dict, target_hw):
        if not bool(self.kd.get('use_teacher_foreground_mask', False)):
            return None

        source = self.kd.get('teacher_foreground_source', 'heatmap').lower()
        pred_key = self.kd.get('teacher_pred_dicts_key', 'teacher_pred_dicts')
        teacher_key = self.kd.get('teacher_feature_key', 'teacher_feature')

        foreground = None
        if source in {'heatmap', 'hm', 'pred'}:
            foreground = self._heatmap_to_foreground(output_dict.get(pred_key, []), target_hw)
        elif source in {'activation', 'feature'}:
            foreground = self._feature_to_foreground(output_dict.get(teacher_key, None))
            if foreground is not None and foreground.shape[-2:] != target_hw:
                foreground = self._resize_mask(foreground, target_hw)
        elif source == 'auto':
            foreground = self._heatmap_to_foreground(output_dict.get(pred_key, []), target_hw)
            if foreground is None:
                foreground = self._feature_to_foreground(output_dict.get(teacher_key, None))
                if foreground is not None and foreground.shape[-2:] != target_hw:
                    foreground = self._resize_mask(foreground, target_hw)
        else:
            raise ValueError(
                f"Unsupported teacher_foreground_source='{source}'. "
                "Use heatmap, activation, or auto."
            )

        return None if foreground is None else foreground.clamp(0.0, 1.0)

    def _build_area_weight_map(self, output_dict, reference, target_hw):
        if not bool(self.kd.get('use_area_rebalance', False)):
            return None

        gt_key = self.kd.get('gt_boxes_key', 'gt_boxes_for_kd')
        gt_boxes = output_dict.get(gt_key, None)
        if gt_boxes is None:
            return None

        gt_boxes = gt_boxes.detach()
        if gt_boxes.ndim != 3 or gt_boxes.shape[-1] < 8:
            return None

        batch_size, _, _ = gt_boxes.shape
        height, width = int(target_hw[0]), int(target_hw[1])
        area_map = reference.new_full((batch_size, 1, height, width), float(self.kd.get('area_background_weight', 0.0)))
        voxel_size = self.kd.get('area_voxel_size', self.kd.get('voxel_size', [0.075, 0.075]))
        lidar_range = self.kd.get('area_lidar_range', self.kd.get('lidar_range', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]))
        stride = float(self.kd.get('area_feature_map_stride', 8.0))
        min_pixels = float(self.kd.get('area_min_pixels', 1.0))
        max_weight = float(self.kd.get('area_max_weight', 10.0))

        vx = float(voxel_size[0]) * stride
        vy = float(voxel_size[1]) * stride
        x_min = float(lidar_range[0])
        y_min = float(lidar_range[1])

        for batch_idx in range(batch_size):
            for box in gt_boxes[batch_idx]:
                cls_id = box[7]
                if cls_id <= 0 or box[3] <= 0 or box[4] <= 0:
                    continue
                cx = int(torch.floor((box[0] - x_min) / vx).item())
                cy = int(torch.floor((box[1] - y_min) / vy).item())
                obj_w = max(int(torch.ceil(box[3] / vx).item()), 1)
                obj_h = max(int(torch.ceil(box[4] / vy).item()), 1)
                x0 = max(cx - obj_w // 2, 0)
                x1 = min(cx + (obj_w + 1) // 2, width)
                y0 = max(cy - obj_h // 2, 0)
                y1 = min(cy + (obj_h + 1) // 2, height)
                if x1 <= x0 or y1 <= y0:
                    continue
                area = max(float((x1 - x0) * (y1 - y0)), min_pixels)
                weight = min(1.0 / area, max_weight)
                patch = area_map[batch_idx, 0, y0:y1, x0:x1]
                area_map[batch_idx, 0, y0:y1, x0:x1] = torch.maximum(patch, patch.new_full(patch.shape, weight))

        if area_map.sum() <= 0:
            return None
        return area_map

    def _build_kd_weight_map(self, output_dict, kd_map):
        mask_key = self.kd.get('motion_mask_key', 'kd_motion_mask')
        use_motion_mask = bool(self.kd.get('use_motion_mask', True))
        use_hybrid_mask = bool(self.kd.get('use_hybrid_mask', False))
        mask_floor = float(self.kd.get('mask_floor', 0.0))
        motion_boost = float(self.kd.get('motion_boost', 0.0))
        teacher_fg_weight = float(self.kd.get('teacher_foreground_weight', 1.0))
        hybrid_mode = self.kd.get('hybrid_mask_mode', 'max').lower()

        motion_mask = None
        if use_motion_mask and mask_key in output_dict:
            motion_mask = output_dict[mask_key].detach().clamp(0.0, 1.0)
            motion_mask = self._resize_mask(motion_mask, kd_map.shape[-2:])

        teacher_fg = self._teacher_foreground_mask(output_dict, kd_map.shape[-2:])

        base_mask = motion_mask
        if use_hybrid_mask and teacher_fg is not None:
            teacher_component = (teacher_fg * teacher_fg_weight).clamp(0.0, 1.0)
            if base_mask is None:
                base_mask = teacher_component
            elif hybrid_mode == 'add':
                base_mask = (base_mask + teacher_component).clamp(0.0, 1.0)
            else:
                base_mask = torch.maximum(base_mask, teacher_component)

        if base_mask is not None:
            kd_weight_map = base_mask * (1.0 - mask_floor) + mask_floor
            if motion_boost > 0.0 and motion_mask is not None:
                kd_weight_map = kd_weight_map * (1.0 + motion_boost * motion_mask)
        else:
            kd_weight_map = kd_map.new_ones(kd_map.shape[0], 1, kd_map.shape[2], kd_map.shape[3])

        rcs_mask_key = self.kd.get('rcs_mask_key', 'rcs_confidence_mask')
        use_rcs_mask = bool(self.kd.get('use_rcs_mask', False))
        rcs_boost = float(self.kd.get('rcs_boost', 0.0))
        if use_rcs_mask and rcs_mask_key in output_dict and rcs_boost > 0.0:
            rcs_mask = output_dict[rcs_mask_key].detach().clamp(0.0, 1.0)
            rcs_mask = self._resize_mask(rcs_mask, kd_map.shape[-2:])
            kd_weight_map = kd_weight_map * (1.0 + rcs_boost * rcs_mask)

        area_map = self._build_area_weight_map(output_dict, kd_map, kd_map.shape[-2:])
        if area_map is not None:
            kd_weight_map = kd_weight_map * area_map

        return kd_weight_map, base_mask, teacher_fg, area_map

    def _weighted_map_mean(self, loss_map, weight_map=None):
        if weight_map is None:
            return loss_map.mean()
        if weight_map.shape[-2:] != loss_map.shape[-2:]:
            weight_map = self._resize_mask(weight_map, loss_map.shape[-2:])
        return (loss_map * weight_map).sum() / weight_map.sum().clamp_min(1e-6)

    def _instance_distill_loss(self, output_dict, kd_weight_map, total_loss):
        hm_weight = float(self.kd.get('instance_hm_weight', 0.0))
        reg_weight = float(self.kd.get('instance_reg_weight', 0.0))
        if hm_weight <= 0.0 and reg_weight <= 0.0:
            zero = total_loss.new_tensor(0.0)
            return zero, zero, zero, 0.0

        student_key = self.kd.get('student_pred_dicts_key', 'radar_pred_dicts')
        teacher_key = self.kd.get('teacher_pred_dicts_key', 'teacher_pred_dicts')
        student_preds = output_dict.get(student_key, [])
        teacher_preds = output_dict.get(teacher_key, [])
        if not student_preds or not teacher_preds or len(student_preds) != len(teacher_preds):
            zero = total_loss.new_tensor(0.0)
            return zero, zero, zero, 0.0

        temperature = max(float(self.kd.get('temperature', 1.0)), 1e-6)
        threshold = float(self.kd.get('teacher_foreground_threshold', 0.10))
        reg_heads = self.kd.get('instance_reg_heads', ['center', 'center_z', 'dim', 'rot', 'vel', 'iou'])
        mask_source = self.kd.get('instance_mask_source', 'teacher_foreground').lower()
        use_kd_weight = bool(self.kd.get('instance_use_kd_weight_map', False))
        hm_losses = []
        reg_losses = []
        mask_means = []

        for student_pred, teacher_pred in zip(student_preds, teacher_preds):
            if 'hm' not in student_pred or 'hm' not in teacher_pred:
                continue
            student_hm = student_pred['hm']
            teacher_hm = teacher_pred['hm'].detach()
            if student_hm.shape != teacher_hm.shape:
                raise ValueError(
                    f"Instance heatmap KD shape mismatch: student {student_hm.shape}, "
                    f"teacher {teacher_hm.shape}"
                )
            teacher_prob = torch.sigmoid(teacher_hm / temperature)
            hm_map = F.binary_cross_entropy_with_logits(
                student_hm / temperature,
                teacher_prob,
                reduction='none',
            ) * (temperature ** 2)
            head_fg = torch.sigmoid(teacher_hm).max(dim=1, keepdim=True)[0]
            head_fg = ((head_fg - threshold) / max(1.0 - threshold, 1e-6)).clamp(0.0, 1.0)

            if mask_source in {'teacher_foreground', 'teacher_fg', 'heatmap'}:
                head_weight = head_fg
            elif mask_source in {'kd', 'kd_weight', 'feature_kd'}:
                head_weight = self._resize_mask(kd_weight_map, head_fg.shape[-2:])
            elif mask_source in {'hybrid', 'max'}:
                head_weight = torch.maximum(head_fg, self._resize_mask(kd_weight_map, head_fg.shape[-2:]))
            else:
                raise ValueError(
                    f"Unsupported instance_mask_source='{mask_source}'. "
                    "Use teacher_foreground, kd, or hybrid."
                )
            if use_kd_weight and mask_source not in {'kd', 'kd_weight', 'feature_kd', 'hybrid', 'max'}:
                head_weight = torch.maximum(head_weight, self._resize_mask(kd_weight_map, head_fg.shape[-2:]))
            head_weight = head_weight.clamp(0.0, 1.0)
            mask_means.append(head_weight.detach().mean())
            hm_losses.append(self._weighted_map_mean(hm_map.mean(dim=1, keepdim=True), head_weight))

            for head_name in reg_heads:
                if head_name not in student_pred or head_name not in teacher_pred:
                    continue
                student_reg = student_pred[head_name]
                teacher_reg = teacher_pred[head_name].detach()
                if student_reg.shape != teacher_reg.shape:
                    raise ValueError(
                        f"Instance {head_name} KD shape mismatch: student {student_reg.shape}, "
                        f"teacher {teacher_reg.shape}"
                    )
                reg_map = F.smooth_l1_loss(student_reg, teacher_reg, reduction='none').mean(dim=1, keepdim=True)
                reg_losses.append(self._weighted_map_mean(reg_map, head_weight))

        hm_loss = torch.stack(hm_losses).mean() * hm_weight if hm_losses else total_loss.new_tensor(0.0)
        reg_loss = torch.stack(reg_losses).mean() * reg_weight if reg_losses else total_loss.new_tensor(0.0)
        mask_mean = torch.stack(mask_means).mean().item() if mask_means else 0.0
        return hm_loss + reg_loss, hm_loss, reg_loss, mask_mean

    def forward(self, output_dict, target_dict, suffix=""):
        if 'loss' not in output_dict:
            raise KeyError(
                "PillarnetFeedbackKdLoss expected output_dict['loss']. "
                "For validation, call the model with batch_data['ego']['compute_loss'] = True."
            )

        total_loss = output_dict['loss']
        self.loss_dict = dict(output_dict.get('tb_dict', {}))

        student_key = self.kd.get('student_feature_key', 'feature')
        teacher_key = self.kd.get('teacher_feature_key', 'teacher_feature')
        kd_weight = float(self.kd.get('weight', 0.0))
        kd_loss_type = self.kd.get('feature_loss', 'mse').lower()
        logit_weight = float(self.kd.get('logit_weight', 0.0))
        temperature = max(float(self.kd.get('temperature', 1.0)), 1e-6)

        kd_loss = total_loss.new_tensor(0.0)
        logit_kd_loss = total_loss.new_tensor(0.0)
        instance_kd_loss = total_loss.new_tensor(0.0)
        instance_hm_loss = total_loss.new_tensor(0.0)
        instance_reg_loss = total_loss.new_tensor(0.0)
        instance_mask_mean = 0.0
        kd_mask_mean = 1.0
        kd_weight_mean = 1.0
        teacher_fg_mean = 0.0
        area_weight_mean = 0.0
        raw_student_rms = 0.0
        raw_teacher_rms = 0.0
        kd_weight_map = None

        if student_key in output_dict and teacher_key in output_dict:
            student_feature_raw = output_dict[student_key]
            teacher_feature_raw = output_dict[teacher_key].detach()
            if student_feature_raw.shape != teacher_feature_raw.shape:
                raise ValueError(
                    f"KD feature shape mismatch: student {student_feature_raw.shape}, "
                    f"teacher {teacher_feature_raw.shape}. Use a PillarNet LiDAR teacher "
                    "with matching model_voxel_size/range/backbone for this student."
                )

            eps = float(self.kd.get('normalization_eps', 1e-6))
            raw_student_rms = self._channel_rms(student_feature_raw.detach(), eps).mean().item()
            raw_teacher_rms = self._channel_rms(teacher_feature_raw.detach(), eps).mean().item()
            student_feature = self._normalize_feature(student_feature_raw)
            teacher_feature = self._normalize_feature(teacher_feature_raw)

            if kd_loss_type == 'smooth_l1':
                kd_map = F.smooth_l1_loss(student_feature, teacher_feature, reduction='none').mean(dim=1, keepdim=True)
            else:
                kd_map = (student_feature - teacher_feature).pow(2).mean(dim=1, keepdim=True)

            kd_weight_map, base_mask, teacher_fg, area_map = self._build_kd_weight_map(output_dict, kd_map)
            kd_mask_mean = self._safe_mean(base_mask) if base_mask is not None else 1.0
            kd_weight_mean = self._safe_mean(kd_weight_map)
            teacher_fg_mean = self._safe_mean(teacher_fg)
            area_weight_mean = self._safe_mean(area_map)

            if kd_weight > 0.0:
                kd_loss = self._weighted_map_mean(kd_map, kd_weight_map) * kd_weight
                total_loss = total_loss + kd_loss

            instance_kd_loss, instance_hm_loss, instance_reg_loss, instance_mask_mean = self._instance_distill_loss(
                output_dict, kd_weight_map, total_loss
            )
            if instance_kd_loss.item() != 0.0:
                total_loss = total_loss + instance_kd_loss

            if logit_weight > 0.0:
                student_cls_key = self.kd.get('student_cls_key', 'cls_preds')
                teacher_cls_key = self.kd.get('teacher_cls_key', 'teacher_cls_preds')
                if student_cls_key in output_dict and teacher_cls_key in output_dict:
                    student_logits = output_dict[student_cls_key]
                    teacher_logits = output_dict[teacher_cls_key].detach()
                    if student_logits.shape != teacher_logits.shape:
                        raise ValueError(
                            f"KD logit shape mismatch: student {student_logits.shape}, "
                            f"teacher {teacher_logits.shape}"
                        )
                    logit_map = F.binary_cross_entropy_with_logits(
                        student_logits / temperature,
                        torch.sigmoid(teacher_logits / temperature),
                        reduction='none',
                    ) * (temperature ** 2)
                    cls_weight = kd_weight_map
                    if cls_weight.shape[-2:] != logit_map.shape[-2:]:
                        cls_weight = self._resize_mask(cls_weight, logit_map.shape[-2:])
                    logit_kd_loss = self._weighted_map_mean(logit_map.mean(dim=1, keepdim=True), cls_weight)
                    logit_kd_loss = logit_kd_loss * logit_weight
                    total_loss = total_loss + logit_kd_loss

        self.loss_dict.update({
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'logit_kd_loss': logit_kd_loss.item(),
            'instance_kd_loss': instance_kd_loss.item(),
            'instance_hm_loss': instance_hm_loss.item(),
            'instance_reg_loss': instance_reg_loss.item(),
            'instance_mask_mean': instance_mask_mean,
            'kd_mask_mean': kd_mask_mean,
            'kd_weight_mean': kd_weight_mean,
            'teacher_fg_mean': teacher_fg_mean,
            'area_weight_mean': area_weight_mean,
            'kd_raw_student_rms': raw_student_rms,
            'kd_raw_teacher_rms': raw_teacher_rms,
        })
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        total_loss = self.loss_dict.get('total_loss', 0.0)
        radar_head_loss = self.loss_dict.get('radar_head_loss', self.loss_dict.get('rpn_loss', 0.0))
        kd_loss = self.loss_dict.get('kd_loss', 0.0)
        logit_kd_loss = self.loss_dict.get('logit_kd_loss', 0.0)
        instance_kd_loss = self.loss_dict.get('instance_kd_loss', 0.0)
        instance_mask_mean = self.loss_dict.get('instance_mask_mean', 0.0)
        kd_mask_mean = self.loss_dict.get('kd_mask_mean', 0.0)
        kd_weight_mean = self.loss_dict.get('kd_weight_mean', 0.0)
        teacher_fg_mean = self.loss_dict.get('teacher_fg_mean', 0.0)
        area_weight_mean = self.loss_dict.get('area_weight_mean', 0.0)
        student_rms = self.loss_dict.get('kd_raw_student_rms', 0.0)
        teacher_rms = self.loss_dict.get('kd_raw_teacher_rms', 0.0)
        print(
            '[epoch %d][%d/%d]%s || Loss: %.4f || Radar Head: %.4f || KD: %.4f || '
            'Inst KD: %.4f || Inst Mask: %.4f || Logit KD: %.4f || KD Mask: %.4f || KD Weight: %.4f || '
            'Teacher FG: %.4f || Area W: %.4f || Raw RMS S/T: %.4f/%.4f'
            % (epoch, batch_id + 1, batch_len, suffix, total_loss, radar_head_loss, kd_loss,
               instance_kd_loss, instance_mask_mean, logit_kd_loss, kd_mask_mean, kd_weight_mean,
               teacher_fg_mean, area_weight_mean, student_rms, teacher_rms)
        )
        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar('Total_loss' + suffix, total_loss, step)
            writer.add_scalar('Radar_head_loss' + suffix, radar_head_loss, step)
            writer.add_scalar('Kd_loss' + suffix, kd_loss, step)
            writer.add_scalar('Logit_kd_loss' + suffix, logit_kd_loss, step)
            writer.add_scalar('Instance_kd_loss' + suffix, instance_kd_loss, step)
            writer.add_scalar('Instance_hm_loss' + suffix, self.loss_dict.get('instance_hm_loss', 0.0), step)
            writer.add_scalar('Instance_reg_loss' + suffix, self.loss_dict.get('instance_reg_loss', 0.0), step)
            writer.add_scalar('Instance_mask_mean' + suffix, instance_mask_mean, step)
            writer.add_scalar('Kd_mask_mean' + suffix, kd_mask_mean, step)
            writer.add_scalar('Kd_weight_mean' + suffix, kd_weight_mean, step)
            writer.add_scalar('Teacher_fg_mean' + suffix, teacher_fg_mean, step)
            writer.add_scalar('Area_weight_mean' + suffix, area_weight_mean, step)
            writer.add_scalar('Kd_raw_student_rms' + suffix, student_rms, step)
            writer.add_scalar('Kd_raw_teacher_rms' + suffix, teacher_rms, step)
