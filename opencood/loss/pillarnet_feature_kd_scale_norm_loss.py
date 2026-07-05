import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarnetFeatureKdScaleNormLoss(nn.Module):
    """PillarNet detection loss + scale-normalized BEV feature KD.

    This variant implements the feature magnitude alignment recommendation for
    radar-to-LiDAR distillation. The student and teacher BEV tensors are
    normalized before the feature-level distillation distance is computed, so
    the KD term compares representation structure instead of raw activation
    scale.
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

    def forward(self, output_dict, target_dict, suffix=""):
        if 'loss' not in output_dict:
            raise KeyError(
                "PillarnetFeatureKdScaleNormLoss expected output_dict['loss']. "
                "For validation, call the model with "
                "batch_data['ego']['compute_loss'] = True."
            )

        total_loss = output_dict['loss']
        self.loss_dict = dict(output_dict.get('tb_dict', {}))

        student_key = self.kd.get('student_feature_key', 'feature')
        teacher_key = self.kd.get('teacher_feature_key', 'teacher_feature')
        mask_key = self.kd.get('motion_mask_key', 'kd_motion_mask')
        kd_weight = float(self.kd.get('weight', 0.0))
        kd_loss_type = self.kd.get('feature_loss', 'mse').lower()
        use_motion_mask = bool(self.kd.get('use_motion_mask', True))
        mask_floor = float(self.kd.get('mask_floor', 0.0))
        motion_boost = float(self.kd.get('motion_boost', 0.0))
        rcs_mask_key = self.kd.get('rcs_mask_key', 'rcs_confidence_mask')
        use_rcs_mask = bool(self.kd.get('use_rcs_mask', False))
        rcs_boost = float(self.kd.get('rcs_boost', 0.0))
        logit_weight = float(self.kd.get('logit_weight', 0.0))
        temperature = max(float(self.kd.get('temperature', 1.0)), 1e-6)

        kd_loss = total_loss.new_tensor(0.0)
        logit_kd_loss = total_loss.new_tensor(0.0)
        kd_mask_mean = 1.0
        kd_weight_mean = 1.0
        raw_student_rms = 0.0
        raw_teacher_rms = 0.0

        if kd_weight > 0.0 and student_key in output_dict and teacher_key in output_dict:
            student_feature = output_dict[student_key]
            teacher_feature = output_dict[teacher_key].detach()
            if student_feature.shape != teacher_feature.shape:
                raise ValueError(
                    f"KD feature shape mismatch: student {student_feature.shape}, "
                    f"teacher {teacher_feature.shape}. Use a PillarNet LiDAR teacher "
                    "with matching model_voxel_size/range/backbone for this student."
                )

            eps = float(self.kd.get('normalization_eps', 1e-6))
            raw_student_rms = self._channel_rms(student_feature.detach(), eps).mean().item()
            raw_teacher_rms = self._channel_rms(teacher_feature.detach(), eps).mean().item()
            student_feature = self._normalize_feature(student_feature)
            teacher_feature = self._normalize_feature(teacher_feature)

            if kd_loss_type == 'smooth_l1':
                kd_map = F.smooth_l1_loss(student_feature, teacher_feature, reduction='none').mean(dim=1, keepdim=True)
            else:
                kd_map = (student_feature - teacher_feature).pow(2).mean(dim=1, keepdim=True)

            if use_motion_mask and mask_key in output_dict:
                motion_mask = output_dict[mask_key].detach().clamp(0.0, 1.0)
                motion_mask = self._resize_mask(motion_mask, kd_map.shape[-2:])
                kd_weight_map = motion_mask * (1.0 - mask_floor) + mask_floor
                if motion_boost > 0.0:
                    kd_weight_map = kd_weight_map * (1.0 + motion_boost * motion_mask)
                kd_mask_mean = motion_mask.mean().item()
            else:
                kd_weight_map = kd_map.new_ones(kd_map.shape[0], 1, kd_map.shape[2], kd_map.shape[3])

            if use_rcs_mask and rcs_mask_key in output_dict and rcs_boost > 0.0:
                rcs_mask = output_dict[rcs_mask_key].detach().clamp(0.0, 1.0)
                rcs_mask = self._resize_mask(rcs_mask, kd_map.shape[-2:])
                kd_weight_map = kd_weight_map * (1.0 + rcs_boost * rcs_mask)

            kd_loss = (kd_map * kd_weight_map).sum() / kd_weight_map.sum().clamp_min(1.0)
            kd_weight_mean = kd_weight_map.mean().item()
            kd_loss = kd_loss * kd_weight
            total_loss = total_loss + kd_loss

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
                    logit_kd_loss = (logit_map.mean(dim=1, keepdim=True) * cls_weight).sum() / cls_weight.sum().clamp_min(1.0)
                    logit_kd_loss = logit_kd_loss * logit_weight
                    total_loss = total_loss + logit_kd_loss

        self.loss_dict.update({
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'logit_kd_loss': logit_kd_loss.item(),
            'kd_mask_mean': kd_mask_mean,
            'kd_weight_mean': kd_weight_mean,
            'kd_raw_student_rms': raw_student_rms,
            'kd_raw_teacher_rms': raw_teacher_rms,
        })
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        total_loss = self.loss_dict.get('total_loss', 0.0)
        radar_head_loss = self.loss_dict.get('radar_head_loss', self.loss_dict.get('rpn_loss', 0.0))
        kd_loss = self.loss_dict.get('kd_loss', 0.0)
        logit_kd_loss = self.loss_dict.get('logit_kd_loss', 0.0)
        kd_mask_mean = self.loss_dict.get('kd_mask_mean', 0.0)
        kd_weight_mean = self.loss_dict.get('kd_weight_mean', 0.0)
        student_rms = self.loss_dict.get('kd_raw_student_rms', 0.0)
        teacher_rms = self.loss_dict.get('kd_raw_teacher_rms', 0.0)
        print(
            '[epoch %d][%d/%d]%s || Loss: %.4f || Radar Head: %.4f || KD: %.4f || '
            'Logit KD: %.4f || KD Mask: %.4f || KD Weight: %.4f || Raw RMS S/T: %.4f/%.4f'
            % (epoch, batch_id + 1, batch_len, suffix, total_loss, radar_head_loss, kd_loss,
               logit_kd_loss, kd_mask_mean, kd_weight_mean, student_rms, teacher_rms)
        )
        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar('Total_loss' + suffix, total_loss, step)
            writer.add_scalar('Radar_head_loss' + suffix, radar_head_loss, step)
            writer.add_scalar('Kd_loss' + suffix, kd_loss, step)
            writer.add_scalar('Logit_kd_loss' + suffix, logit_kd_loss, step)
            writer.add_scalar('Kd_mask_mean' + suffix, kd_mask_mean, step)
            writer.add_scalar('Kd_weight_mean' + suffix, kd_weight_mean, step)
            writer.add_scalar('Kd_raw_student_rms' + suffix, student_rms, step)
            writer.add_scalar('Kd_raw_teacher_rms' + suffix, teacher_rms, step)
