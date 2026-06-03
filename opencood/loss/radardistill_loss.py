import torch.nn as nn


class RadardistillLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=''):
        self.loss_dict = output_dict.get('tb_dict', {})
        if 'loss' not in output_dict:
            raise KeyError(
                "RadardistillLoss expected output_dict['loss'], but the model "
                "did not compute a loss. During validation call the model with "
                "batch_data['ego']['compute_loss'] = True before forward()."
            )
        return output_dict['loss']

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=''):
        total_loss = self.loss_dict.get('total_loss', 0.0)
        teacher_head_loss = self.loss_dict.get('teacher_head_loss', 0.0)
        radar_head_loss = self.loss_dict.get('radar_head_loss', 0.0)
        distill_loss = self.loss_dict.get('distill_loss', 0.0)
        print(
            '[epoch %d][%d/%d]%s || Loss: %.4f || Teacher Head: %.4f || Radar Head: %.4f || Distill: %.4f'
            % (epoch, batch_id + 1, batch_len, suffix, total_loss, teacher_head_loss, radar_head_loss, distill_loss)
        )
        if writer is not None:
            writer.add_scalar('Total_loss' + suffix, total_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Teacher_head_loss' + suffix, teacher_head_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Radar_head_loss' + suffix, radar_head_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Distill_loss' + suffix, distill_loss, epoch * batch_len + batch_id)