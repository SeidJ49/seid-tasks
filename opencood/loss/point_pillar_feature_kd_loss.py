import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.point_pillar_loss import PointPillarLoss


class PointPillarFeatureKdLoss(PointPillarLoss):
    def __init__(self, args):
        super().__init__(args)
        self.kd = args['kd']

    def forward(self, output_dict, target_dict, suffix=""):
        total_loss = super().forward(output_dict, target_dict, suffix=suffix)

        student_key = self.kd.get('student_feature_key', 'feature')
        teacher_key = self.kd.get('teacher_feature_key', 'teacher_feature')
        mask_key = self.kd.get('motion_mask_key', 'kd_motion_mask')
        kd_loss_type = self.kd.get('feature_loss', 'mse').lower()
        kd_weight = float(self.kd.get('weight', 1.0))
        use_motion_mask = bool(self.kd.get('use_motion_mask', True))
        mask_floor = float(self.kd.get('mask_floor', 0.0))
        motion_boost = float(self.kd.get('motion_boost', 0.0))
        logit_weight = float(self.kd.get('logit_weight', 0.0))
        temperature = max(float(self.kd.get('temperature', 1.0)), 1e-6)

        assert student_key in output_dict, f"Missing student feature key: {student_key}"
        assert teacher_key in output_dict, f"Missing teacher feature key: {teacher_key}"

        student_feature = output_dict[student_key]
        teacher_feature = output_dict[teacher_key].detach()

        if student_feature.shape != teacher_feature.shape:
            raise ValueError(
                f"KD feature shape mismatch: student {student_feature.shape}, teacher {teacher_feature.shape}"
            )

        if kd_loss_type == 'smooth_l1':
            kd_map = nn.functional.smooth_l1_loss(
                student_feature,
                teacher_feature,
                reduction='none'
            ).mean(dim=1, keepdim=True)
        else:
            kd_map = (student_feature - teacher_feature).pow(2).mean(dim=1, keepdim=True)

        kd_weight_map = None
        if use_motion_mask and mask_key in output_dict:
            motion_mask = output_dict[mask_key].detach()
            if motion_mask.shape[-2:] != kd_map.shape[-2:]:
                motion_mask = nn.functional.interpolate(
                    motion_mask,
                    size=kd_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            motion_mask = motion_mask.clamp(0.0, 1.0)

            # RadarDistill uses activation/proposal masks to avoid treating all
            # BEV cells equally. For this thesis we keep the supervision
            # motion-aware instead: every cell can still learn from LiDAR via
            # mask_floor, while Doppler-active radar regions receive an
            # additional boost. This preserves the thesis focus on radar motion
            # cues instead of blindly copying LiDAR everywhere.
            kd_weight_map = motion_mask * (1.0 - mask_floor) + mask_floor
            if motion_boost > 0.0:
                kd_weight_map = kd_weight_map * (1.0 + motion_boost * motion_mask)
            kd_loss = (kd_map * kd_weight_map).sum() / kd_weight_map.sum().clamp_min(1.0)
            self.loss_dict['kd_mask_mean'] = motion_mask.mean().item()
            self.loss_dict['kd_weight_mean'] = kd_weight_map.mean().item()
        else:
            kd_loss = kd_map.mean()
            self.loss_dict['kd_mask_mean'] = 1.0
            self.loss_dict['kd_weight_mean'] = 1.0

        kd_loss = kd_loss * kd_weight
        total_loss = total_loss + kd_loss

        logit_kd_loss = student_feature.new_tensor(0.0)
        if logit_weight > 0.0:
            student_cls_key = self.kd.get('student_cls_key', 'cls_preds')
            teacher_cls_key = self.kd.get('teacher_cls_key', 'teacher_cls_preds')
            if student_cls_key in output_dict and teacher_cls_key in output_dict:
                student_logits = output_dict[student_cls_key]
                teacher_logits = output_dict[teacher_cls_key].detach()
                if student_logits.shape != teacher_logits.shape:
                    raise ValueError(
                        f"KD logit shape mismatch: student {student_logits.shape}, teacher {teacher_logits.shape}"
                    )
                logit_map = F.binary_cross_entropy_with_logits(
                    student_logits / temperature,
                    torch.sigmoid(teacher_logits / temperature),
                    reduction='none',
                ) * (temperature ** 2)
                if kd_weight_map is not None:
                    cls_weight = kd_weight_map
                    if cls_weight.shape[-2:] != logit_map.shape[-2:]:
                        cls_weight = F.interpolate(
                            cls_weight,
                            size=logit_map.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    logit_kd_loss = (logit_map.mean(dim=1, keepdim=True) * cls_weight).sum() / cls_weight.sum().clamp_min(1.0)
                else:
                    logit_kd_loss = logit_map.mean()
                logit_kd_loss = logit_kd_loss * logit_weight
                total_loss = total_loss + logit_kd_loss

        self.loss_dict.update({
            'kd_loss': kd_loss.item(),
            'logit_kd_loss': logit_kd_loss.item(),
            'total_loss': total_loss.item(),
        })

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        super().logging(epoch, batch_id, batch_len, writer=writer, suffix=suffix)
        kd_loss = self.loss_dict.get('kd_loss', 0)
        logit_kd_loss = self.loss_dict.get('logit_kd_loss', 0)
        kd_mask_mean = self.loss_dict.get('kd_mask_mean', 0)
        kd_weight_mean = self.loss_dict.get('kd_weight_mean', 0)
        print("[epoch %d][%d/%d]%s || KD Loss: %.4f || Logit KD: %.4f || KD Mask Mean: %.4f || KD Weight Mean: %.4f" % (
            epoch, batch_id + 1, batch_len, suffix, kd_loss, logit_kd_loss, kd_mask_mean, kd_weight_mean
        ))

        if writer is not None:
            writer.add_scalar('Kd_loss' + suffix, kd_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Logit_kd_loss' + suffix, logit_kd_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Kd_mask_mean' + suffix, kd_mask_mean, epoch * batch_len + batch_id)
            writer.add_scalar('Kd_weight_mean' + suffix, kd_weight_mean, epoch * batch_len + batch_id)