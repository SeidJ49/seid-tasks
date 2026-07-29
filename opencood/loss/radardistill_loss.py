import torch
import torch.nn as nn


class RadardistillLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_dict = {}
        self.image_dict = {}
        self.image_log_interval = int(args.get('image_log_interval', 100))
        self.image_log_overwrite = bool(args.get('image_log_overwrite', False))

    @staticmethod
    def _tb_tag(name, suffix=''):
        split = 'train'
        if suffix:
            split = suffix.strip('_') or 'train'
        return '%s/%s' % (split, name)

    def forward(self, output_dict, target_dict, suffix=''):
        self.loss_dict = output_dict.get('tb_dict', {})
        self.image_dict = output_dict.get('debug_maps', {})
        if 'loss' not in output_dict:
            raise KeyError(
                "RadardistillLoss expected output_dict['loss'], but the model "
                "did not compute a loss. During validation call the model with "
                "batch_data['ego']['compute_loss'] = True before forward()."
            )
        return output_dict['loss']

    @staticmethod
    def _first_mask_image(mask):
        if mask is None or not torch.is_tensor(mask):
            return None
        image = mask.detach()
        if image.ndim == 4:
            image = image[0]
        elif image.ndim == 3:
            pass
        elif image.ndim == 2:
            image = image.unsqueeze(0)
        else:
            return None
        if image.shape[0] != 1 and image.shape[0] != 3:
            image = image[:1]
        image = image.float().cpu()
        # Masks are already probabilities/weights in [0, 1]. Keep absolute
        # scaling so TensorBoard brightness is comparable across mask types.
        return image.clamp(0.0, 1.0)

    def _log_mask_images(self, writer, suffix, step):
        if not self.image_dict or self.image_log_interval <= 0:
            return
        if step % self.image_log_interval != 0:
            return
        image_step = 0 if self.image_log_overwrite else step

        image_keys = {
            'teacher_heatmap_mask': 'images/teacher_heatmap_mask',
            'gt_box_mask': 'images/gt_box_mask',
            'gt_box_outline_mask': 'images/gt_box_outline_mask',
            'object_kd_mask': 'images/object_kd_mask',
            'radar_evidence_mask': 'images/radar_evidence_mask',
            'radar_doppler_score_mask': 'images/radar_doppler_score_mask',
            'radar_rcs_confidence_mask': 'images/radar_rcs_confidence_mask',
            'radar_rcs_mid_support_mask': 'images/radar_rcs_mid_support_mask',
            'radar_rcs_high_support_mask': 'images/radar_rcs_high_support_mask',
            'lidar_intensity_mask': 'images/lidar_intensity_mask',
            'object_lidar_intensity_gate_mask': 'images/object_lidar_intensity_gate_mask',
            'object_radar_evidence_gate_mask': 'images/object_radar_evidence_gate_mask',
            'distill_final_mask': 'images/distill_final_mask',
        }
        for key, tag in image_keys.items():
            image = self._first_mask_image(self.image_dict.get(key))
            if image is not None:
                writer.add_image(self._tb_tag(tag, suffix), image, image_step)

        radar = self._first_mask_image(self.image_dict.get('radar_evidence_mask'))
        obj = self._first_mask_image(self.image_dict.get('object_kd_mask'))
        lidar = self._first_mask_image(self.image_dict.get('lidar_intensity_mask'))
        if radar is not None and obj is not None and lidar is not None:
            overlay = torch.cat([radar[:1], obj[:1], lidar[:1]], dim=0)
            writer.add_image(self._tb_tag('images/rgb_radar_object_lidar', suffix), overlay, image_step)

        object_radar = self._first_mask_image(self.image_dict.get('object_radar_evidence_gate_mask'))
        box = self._first_mask_image(self.image_dict.get('gt_box_mask'))
        object_lidar = self._first_mask_image(self.image_dict.get('object_lidar_intensity_gate_mask'))
        if object_radar is not None and box is not None and object_lidar is not None:
            overlay = torch.cat([object_radar[:1], box[:1], object_lidar[:1]], dim=0)
            writer.add_image(self._tb_tag('images/rgb_object_radar_box_lidar', suffix), overlay, image_step)

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=''):
        total_loss = self.loss_dict.get('total_loss', 0.0)
        teacher_head_loss = self.loss_dict.get('teacher_head_loss', 0.0)
        radar_head_loss = self.loss_dict.get('radar_head_loss', self.loss_dict.get('rpn_loss', 0.0))
        distill_loss = self.loss_dict.get('distill_loss', 0.0)
        log_msg = (
            '[epoch %d][%d/%d]%s || Loss: %.4f || Teacher Head: %.4f || Radar Head: %.4f || Distill: %.4f'
            % (epoch, batch_id + 1, batch_len, suffix, total_loss, teacher_head_loss, radar_head_loss, distill_loss)
        )
        if 'radar_evidence_mask_mean' in self.loss_dict:
            log_msg += ' || Radar Evidence Mask: %.4f' % self.loss_dict['radar_evidence_mask_mean']
        if 'lidar_intensity_mask_mean' in self.loss_dict:
            log_msg += ' || LiDAR Intensity Mask: %.4f' % self.loss_dict['lidar_intensity_mask_mean']
        if 'distill_final_mask_mean' in self.loss_dict:
            log_msg += ' || Distill Mask: %.4f' % self.loss_dict['distill_final_mask_mean']
        if 'object_kd_loss' in self.loss_dict:
            log_msg += ' || Object KD: %.4f' % self.loss_dict['object_kd_loss']
        if 'object_kd_mask_mean' in self.loss_dict:
            log_msg += ' || Object KD Mask: %.4f' % self.loss_dict['object_kd_mask_mean']
        print(log_msg)
        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar(self._tb_tag('loss/total', suffix), total_loss, step)
            writer.add_scalar(self._tb_tag('loss/teacher_head', suffix), teacher_head_loss, step)
            writer.add_scalar(self._tb_tag('loss/radar_head', suffix), radar_head_loss, step)
            writer.add_scalar(self._tb_tag('distill/total', suffix), distill_loss, step)
            writer.add_scalar(
                self._tb_tag('distill/low_feature', suffix),
                self.loss_dict.get('low_feature_loss', 0.0),
                step,
            )
            writer.add_scalar(
                self._tb_tag('distill/low_feature_de8x', suffix),
                self.loss_dict.get('low_feature_loss_de8x', 0.0),
                step,
            )
            writer.add_scalar(
                self._tb_tag('distill/high_feature', suffix),
                self.loss_dict.get('high_distill_loss', 0.0),
                step,
            )
            writer.add_scalar(
                self._tb_tag('distill/afd_mask', suffix),
                self.loss_dict.get('mask_loss', 0.0),
                step,
            )
            writer.add_scalar(
                self._tb_tag('distill/afd_mask_de8x', suffix),
                self.loss_dict.get('mask_loss_de8x', 0.0),
                step,
            )
            if 'radar_evidence_mask_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/radar_evidence_mean', suffix),
                    self.loss_dict['radar_evidence_mask_mean'],
                    step,
                )
            if 'lidar_intensity_mask_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/lidar_intensity_mean', suffix),
                    self.loss_dict['lidar_intensity_mask_mean'],
                    step,
                )
            if 'distill_final_mask_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/distill_final_mean', suffix),
                    self.loss_dict['distill_final_mask_mean'],
                    step,
                )
            if 'object_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/object_kd', suffix),
                    self.loss_dict['object_kd_loss'],
                    step,
                )
            if 'object_kd_mask_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/object_kd_mean', suffix),
                    self.loss_dict['object_kd_mask_mean'],
                    step,
                )
            self._log_mask_images(writer, suffix, step)
