import torch
import torch.nn as nn


class RadardistillLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_dict = {}
        self.image_dict = {}
        self.image_log_interval = int(args.get('image_log_interval', 100))
        self.image_log_overwrite = bool(args.get('image_log_overwrite', False))
        self.image_log_detail = bool(args.get('image_log_detail', False))

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

        compact_image_keys = {
            'radar_evidence_mask': 'images/compact/radar_evidence_mask',
            'lidar_intensity_mask': 'images/compact/lidar_intensity_mask',
            'gt_box_mask': 'images/compact/gt_box_mask',
            'semantic_heatmap_delta_mask': 'images/compact/semantic_heatmap_delta_mask',
            'proposal_fn_mask': 'images/compact/proposal_fn_mask',
            'proposal_fp_mask': 'images/compact/proposal_fp_mask',
            'masked_low_main_afd_weight_mask': 'images/compact/masked_low_main_afd_weight_mask',
            'masked_low_main_physical_mask': 'images/compact/masked_low_main_physical_mask',
            'task_dense_weight_mask': 'images/compact/task_dense_weight_mask',
            'task_dense_gt_box_mask': 'images/compact/task_dense_gt_box_mask',
            'task_dense_feature_delta_mask': 'images/compact/task_dense_feature_delta_mask',
            'response_pos_weight_mask': 'images/compact/response_pos_weight_mask',
            'response_neg_fp_mask': 'images/compact/response_neg_fp_mask',
            'learned_mask_prior': 'images/compact/learned_mask_prior',
            'learned_mask_pred': 'images/compact/learned_mask_pred',
            'learned_mask_kd_weight': 'images/compact/learned_mask_kd_weight',
        }
        detailed_image_keys = {
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
            'object_proto_radar_support_mask': 'images/object_proto_radar_support_mask',
            'object_proto_context_mask': 'images/object_proto_context_mask',
            'semantic_teacher_heatmap_mask': 'images/semantic_teacher_heatmap_mask',
            'semantic_radar_heatmap_mask': 'images/semantic_radar_heatmap_mask',
            'semantic_gt_heatmap_mask': 'images/semantic_gt_heatmap_mask',
            'semantic_heatmap_weight_mask': 'images/semantic_heatmap_weight_mask',
            'semantic_heatmap_delta_mask': 'images/semantic_heatmap_delta_mask',
            'proposal_fn_mask': 'images/proposal_fn_mask',
            'proposal_fp_mask': 'images/proposal_fp_mask',
            'distill_final_mask': 'images/distill_final_mask',
            'masked_low_main_afd_lidar_active_mask': 'images/masked_low_main_afd_lidar_active_mask',
            'masked_low_main_afd_radar_active_mask': 'images/masked_low_main_afd_radar_active_mask',
            'masked_low_main_afd_overlap_mask': 'images/masked_low_main_afd_overlap_mask',
            'masked_low_main_afd_radar_only_mask': 'images/masked_low_main_afd_radar_only_mask',
            'masked_low_main_afd_weight_mask': 'images/masked_low_main_afd_weight_mask',
            'masked_low_main_physical_mask': 'images/masked_low_main_physical_mask',
            'masked_low_de8x_afd_weight_mask': 'images/masked_low_de8x_afd_weight_mask',
            'masked_low_de8x_physical_mask': 'images/masked_low_de8x_physical_mask',
            'task_dense_gt_mask': 'images/task_dense_gt_mask',
            'task_dense_gt_heatmap_mask': 'images/task_dense_gt_heatmap_mask',
            'task_dense_gt_box_mask': 'images/task_dense_gt_box_mask',
            'task_dense_teacher_conf_mask': 'images/task_dense_teacher_conf_mask',
            'task_dense_reliability_mask': 'images/task_dense_reliability_mask',
            'task_dense_weight_mask': 'images/task_dense_weight_mask',
            'task_dense_feature_delta_mask': 'images/task_dense_feature_delta_mask',
            'response_pos_weight_mask': 'images/response_pos_weight_mask',
            'response_neg_fp_mask': 'images/response_neg_fp_mask',
            'response_heatmap_delta_mask': 'images/response_heatmap_delta_mask',
            'learned_mask_prior': 'images/learned_mask_prior',
            'learned_mask_prior_box': 'images/learned_mask_prior_box',
            'learned_mask_prior_teacher': 'images/learned_mask_prior_teacher',
            'learned_mask_pred': 'images/learned_mask_pred',
            'learned_mask_kd_weight': 'images/learned_mask_kd_weight',
            'learned_mask_feature_delta': 'images/learned_mask_feature_delta',
        }
        image_keys = detailed_image_keys if self.image_log_detail else compact_image_keys
        for key, tag in image_keys.items():
            image = self._first_mask_image(self.image_dict.get(key))
            if image is not None:
                writer.add_image(self._tb_tag(tag, suffix), image, image_step)

        radar = self._first_mask_image(self.image_dict.get('radar_evidence_mask'))
        obj = self._first_mask_image(self.image_dict.get('object_kd_mask'))
        lidar = self._first_mask_image(self.image_dict.get('lidar_intensity_mask'))
        if radar is not None and obj is not None and lidar is not None:
            overlay = torch.cat([radar[:1], obj[:1], lidar[:1]], dim=0)
            tag = 'images/rgb_radar_object_lidar' if self.image_log_detail else 'images/compact/rgb_radar_object_lidar'
            writer.add_image(self._tb_tag(tag, suffix), overlay, image_step)

        object_radar = self._first_mask_image(self.image_dict.get('object_radar_evidence_gate_mask'))
        box = self._first_mask_image(self.image_dict.get('gt_box_mask'))
        object_lidar = self._first_mask_image(self.image_dict.get('object_lidar_intensity_gate_mask'))
        if self.image_log_detail and object_radar is not None and box is not None and object_lidar is not None:
            overlay = torch.cat([object_radar[:1], box[:1], object_lidar[:1]], dim=0)
            writer.add_image(self._tb_tag('images/rgb_object_radar_box_lidar', suffix), overlay, image_step)

        proto_support = self._first_mask_image(self.image_dict.get('object_proto_radar_support_mask'))
        proto_context = self._first_mask_image(self.image_dict.get('object_proto_context_mask'))
        if radar is not None and proto_support is not None and proto_context is not None:
            overlay = torch.cat([radar[:1], proto_support[:1], proto_context[:1]], dim=0)
            tag = 'images/rgb_radar_proto_context' if self.image_log_detail else 'images/compact/rgb_radar_proto_context'
            writer.add_image(self._tb_tag(tag, suffix), overlay, image_step)

        sem_radar = self._first_mask_image(self.image_dict.get('semantic_radar_heatmap_mask'))
        sem_teacher = self._first_mask_image(self.image_dict.get('semantic_teacher_heatmap_mask'))
        sem_weight = self._first_mask_image(self.image_dict.get('semantic_heatmap_weight_mask'))
        if sem_radar is not None and sem_teacher is not None and sem_weight is not None:
            overlay = torch.cat([sem_radar[:1], sem_teacher[:1], sem_weight[:1]], dim=0)
            tag = (
                'images/rgb_semantic_radar_teacher_weight'
                if self.image_log_detail else
                'images/compact/rgb_semantic_radar_teacher_weight'
            )
            writer.add_image(self._tb_tag(tag, suffix), overlay, image_step)

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
        if 'object_proto_kd_loss' in self.loss_dict:
            log_msg += ' || Proto KD: %.4f' % self.loss_dict['object_proto_kd_loss']
        if 'semantic_heatmap_kd_loss' in self.loss_dict:
            log_msg += ' || Sem KD: %.4f' % self.loss_dict['semantic_heatmap_kd_loss']
        if 'task_dense_feature_kd_loss' in self.loss_dict:
            log_msg += ' || Task Dense KD: %.4f' % self.loss_dict['task_dense_feature_kd_loss']
        if 'response_kd_loss' in self.loss_dict:
            log_msg += ' || Response KD: %.4f' % self.loss_dict['response_kd_loss']
        if 'learned_mask_loss' in self.loss_dict:
            log_msg += ' || Learned Mask: %.4f' % self.loss_dict['learned_mask_loss']
        if 'proposal_error_kd_loss' in self.loss_dict:
            log_msg += ' || Proposal Error KD: %.4f' % self.loss_dict['proposal_error_kd_loss']
        if 'proposal_fp_cell_mean' in self.loss_dict:
            log_msg += ' || FP Cells: %.4f' % self.loss_dict['proposal_fp_cell_mean']
        if 'proposal_fn_cell_mean' in self.loss_dict:
            log_msg += ' || FN Cells: %.4f' % self.loss_dict['proposal_fn_cell_mean']
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
            if 'object_proto_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/object_proto_kd', suffix),
                    self.loss_dict['object_proto_kd_loss'],
                    step,
                )
            if 'object_proto_observability_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/object_proto_observability_mean', suffix),
                    self.loss_dict['object_proto_observability_mean'],
                    step,
                )
            if 'object_proto_mask_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/object_proto_mean', suffix),
                    self.loss_dict['object_proto_mask_mean'],
                    step,
                )
            if 'semantic_heatmap_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/semantic_heatmap_kd', suffix),
                    self.loss_dict['semantic_heatmap_kd_loss'],
                    step,
                )
            if 'semantic_heatmap_weight_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/semantic_heatmap_weight_mean', suffix),
                    self.loss_dict['semantic_heatmap_weight_mean'],
                    step,
                )
            if 'task_dense_feature_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/task_dense_feature_kd', suffix),
                    self.loss_dict['task_dense_feature_kd_loss'],
                    step,
                )
            if 'task_dense_mask_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/task_dense_mean', suffix),
                    self.loss_dict['task_dense_mask_mean'],
                    step,
                )
            if 'task_dense_teacher_conf_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/task_dense_teacher_conf_mean', suffix),
                    self.loss_dict['task_dense_teacher_conf_mean'],
                    step,
                )
            if 'response_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/response_kd', suffix),
                    self.loss_dict['response_kd_loss'],
                    step,
                )
            if 'response_pos_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/response_pos_kd', suffix),
                    self.loss_dict['response_pos_kd_loss'],
                    step,
                )
            if 'response_neg_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/response_neg_kd', suffix),
                    self.loss_dict['response_neg_kd_loss'],
                    step,
                )
            if 'response_pos_weight_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/response_pos_weight_mean', suffix),
                    self.loss_dict['response_pos_weight_mean'],
                    step,
                )
            if 'response_neg_cell_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/response_neg_cell_mean', suffix),
                    self.loss_dict['response_neg_cell_mean'],
                    step,
                )
            if 'learned_mask_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/learned_mask_loss', suffix),
                    self.loss_dict['learned_mask_loss'],
                    step,
                )
            if 'learned_mask_feature_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/learned_mask_feature_kd', suffix),
                    self.loss_dict['learned_mask_feature_kd_loss'],
                    step,
                )
            if 'learned_mask_prior_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/learned_prior_mean', suffix),
                    self.loss_dict['learned_mask_prior_mean'],
                    step,
                )
            if 'learned_mask_pred_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/learned_pred_mean', suffix),
                    self.loss_dict['learned_mask_pred_mean'],
                    step,
                )
            if 'learned_mask_kd_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('mask/learned_kd_mean', suffix),
                    self.loss_dict['learned_mask_kd_mean'],
                    step,
                )
            if 'proposal_error_kd_loss' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('distill/proposal_error_kd', suffix),
                    self.loss_dict['proposal_error_kd_loss'],
                    step,
                )
            if 'proposal_fp_cell_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('proposal/fp_cell_mean', suffix),
                    self.loss_dict['proposal_fp_cell_mean'],
                    step,
                )
            if 'proposal_fn_cell_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('proposal/fn_cell_mean', suffix),
                    self.loss_dict['proposal_fn_cell_mean'],
                    step,
                )
            if 'proposal_fp_weight_mean' in self.loss_dict:
                writer.add_scalar(
                    self._tb_tag('proposal/fp_weight_mean', suffix),
                    self.loss_dict['proposal_fp_weight_mean'],
                    step,
                )
            self._log_mask_images(writer, suffix, step)
