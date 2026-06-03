import torch.nn as nn

from opencood.loss.point_pillar_loss import PointPillarLoss


class PointPillarFeatureKdHardmaskLoss(PointPillarLoss):
    """
    Detection loss + masked BEV feature KD with explicit binary motion mask.
    """

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

        if use_motion_mask and mask_key in output_dict:
            motion_mask = output_dict[mask_key].detach()
            if motion_mask.shape[-2:] != kd_map.shape[-2:]:
                motion_mask = nn.functional.interpolate(
                    motion_mask,
                    size=kd_map.shape[-2:],
                    mode='nearest',
                )
            motion_mask = (motion_mask > 0).float()
            kd_loss = (kd_map * motion_mask).sum() / motion_mask.sum().clamp_min(1.0)
            self.loss_dict['kd_mask_mean'] = motion_mask.mean().item()
            self.loss_dict['kd_active_cells'] = motion_mask.sum().item()
        else:
            kd_loss = kd_map.mean()
            self.loss_dict['kd_mask_mean'] = 1.0
            self.loss_dict['kd_active_cells'] = float(kd_map.shape[0] * kd_map.shape[-1] * kd_map.shape[-2])

        kd_loss = kd_loss * kd_weight
        total_loss = total_loss + kd_loss

        self.loss_dict.update({
            'kd_loss': kd_loss.item(),
            'total_loss': total_loss.item(),
        })

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        super().logging(epoch, batch_id, batch_len, writer=writer, suffix=suffix)
        kd_loss = self.loss_dict.get('kd_loss', 0)
        kd_mask_mean = self.loss_dict.get('kd_mask_mean', 0)
        kd_active_cells = self.loss_dict.get('kd_active_cells', 0)
        print("[epoch %d][%d/%d]%s || KD Loss: %.4f || KD Mask Mean: %.4f || KD Active: %.1f" % (
            epoch, batch_id + 1, batch_len, suffix, kd_loss, kd_mask_mean, kd_active_cells
        ))

        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar('Kd_loss' + suffix, kd_loss, step)
            writer.add_scalar('Kd_mask_mean' + suffix, kd_mask_mean, step)
            writer.add_scalar('Kd_active_cells' + suffix, kd_active_cells, step)
