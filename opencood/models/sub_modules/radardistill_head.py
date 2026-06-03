import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import nms_gpu


def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)


def gaussian_radius(height, width, min_overlap=0.5):
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    return torch.min(torch.min(r1, r2), r3)


def gaussian2d(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    heatmap = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
    return heatmap


def draw_gaussian_to_heatmap(heatmap, center, radius):
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian, out=masked_heatmap)


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


def _topk(scores, topk=40):
    batch, num_class, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), topk)
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode='trunc').float()
    topk_xs = (topk_inds % width).float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
    topk_classes = torch.div(topk_ind, topk, rounding_mode='trunc').int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, topk)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, topk)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, topk)
    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = box_scores >= score_thresh
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = box_scores.new_zeros((0,), dtype=torch.long)
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(
            box_scores, k=min(nms_config['nms_pre_maxsize'], box_scores.shape[0])
        )
        boxes_for_nms = box_preds[indices]
        keep_idx, _ = nms_gpu(
            boxes_for_nms[:, 0:7],
            box_scores_nms,
            nms_config['nms_thresh'],
            pre_maxsize=nms_config['nms_pre_maxsize'],
        )
        selected = indices[keep_idx[:nms_config['nms_post_maxsize']]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero(as_tuple=False).view(-1)
        selected = original_idxs[selected]

    return selected, src_box_scores[selected]


def decode_bbox_from_heatmap(
    heatmap,
    rot_cos,
    rot_sin,
    center,
    center_z,
    dim,
    iou=None,
    rectifier=0.0,
    point_cloud_range=None,
    voxel_size=None,
    feature_map_stride=None,
    vel=None,
    topk=100,
    score_thresh=None,
    post_center_limit_range=None,
):
    batch_size, _, _, _ = heatmap.size()
    scores, inds, class_ids, ys, xs = _topk(heatmap, topk=topk)
    center = _transpose_and_gather_feat(center, inds).view(batch_size, topk, 2)
    rot_sin = _transpose_and_gather_feat(rot_sin, inds).view(batch_size, topk, 1)
    rot_cos = _transpose_and_gather_feat(rot_cos, inds).view(batch_size, topk, 1)
    center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, topk, 1)
    dim = _transpose_and_gather_feat(dim, inds).view(batch_size, topk, 3)
    if iou is not None:
        iou = _transpose_and_gather_feat(iou, inds).view(batch_size, topk, 1)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = xs.view(batch_size, topk, 1) + center[:, :, 0:1]
    ys = ys.view(batch_size, topk, 1) + center[:, :, 1:2]
    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_parts = [xs, ys, center_z, dim, angle]
    if vel is not None:
        vel = _transpose_and_gather_feat(vel, inds).view(batch_size, topk, 2)
        box_parts.append(vel)

    final_box_preds = torch.cat(box_parts, dim=-1)
    final_scores = scores.view(batch_size, topk)
    final_class_ids = class_ids.view(batch_size, topk)

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(dim=2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(dim=2)
    if score_thresh is not None:
        mask &= final_scores > score_thresh

    pred_dicts = []
    for batch_index in range(batch_size):
        cur_mask = mask[batch_index]
        cur_boxes = final_box_preds[batch_index, cur_mask]
        cur_scores = final_scores[batch_index, cur_mask]
        cur_labels = final_class_ids[batch_index, cur_mask]
        if iou is not None:
            cur_iou = torch.clamp(iou[batch_index, cur_mask].view(-1), min=0.0, max=1.0)
            cur_scores = torch.pow(cur_scores, 1 - rectifier) * torch.pow(cur_iou, rectifier)
        pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels,
        })

    return pred_dicts


def center_to_corner2d(center, dim):
    corners_norm = torch.tensor(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=torch.float32,
        device=dim.device,
    )
    corners = dim.view(-1, 1, 2) * corners_norm.view(1, 4, 2)
    return corners + center.view(-1, 1, 2)


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    pred_corners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gt_corners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(pred_corners[:, 2], gt_corners[:, 2])
    inter_min_xy = torch.maximum(pred_corners[:, 0], gt_corners[:, 0])
    out_max_xy = torch.maximum(pred_corners[:, 2], gt_corners[:, 2])
    out_min_xy = torch.minimum(pred_corners[:, 0], gt_corners[:, 0])

    volume_pred = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5],
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5],
    ) - torch.maximum(
        pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5],
        gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5],
    )
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp(inter_max_xy - inter_min_xy, min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt + volume_pred - volume_inter

    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(dim=-1)
    outer_h = torch.maximum(
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5],
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5],
    ) - torch.minimum(
        gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5],
        pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5],
    )
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp(out_max_xy - out_min_xy, min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    return torch.clamp(dious, min=-1.0, max=1.0)


def bbox3d_overlaps_giou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    pred_corners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gt_corners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(pred_corners[:, 2], gt_corners[:, 2])
    inter_min_xy = torch.maximum(pred_corners[:, 0], gt_corners[:, 0])
    out_max_xy = torch.maximum(pred_corners[:, 2], gt_corners[:, 2])
    out_min_xy = torch.minimum(pred_corners[:, 0], gt_corners[:, 0])

    volume_pred = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5],
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5],
    ) - torch.maximum(
        gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5],
        pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5],
    )
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp(inter_max_xy - inter_min_xy, min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt + volume_pred - volume_inter

    outer_h = torch.maximum(
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5],
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5],
    ) - torch.minimum(
        gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5],
        pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5],
    )
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp(out_max_xy - out_min_xy, min=0)
    closure = outer[:, 0] * outer[:, 1] * outer_h

    gious = volume_inter / volume_union - (closure - volume_union) / closure
    return torch.clamp(gious, min=-1.0, max=1.0)


class FocalLossCenterNet(nn.Module):
    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


def _reg_loss(pred, gt, mask):
    num = mask.float().sum()
    gt = torch.where(torch.isnan(gt), pred, gt)
    mask = mask.unsqueeze(2).expand_as(gt).float()
    loss = torch.abs(pred * mask - gt * mask)
    loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
    return loss / torch.clamp_min(num, min=1.0)


class RegLossCenterNet(nn.Module):
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        return _reg_loss(pred, target, mask)


class IouLossCenterNet(nn.Module):
    def forward(self, iou_pred, mask, ind, pred_boxes, gt_boxes):
        pred_iou = _transpose_and_gather_feat(iou_pred, ind).squeeze(-1)
        pred_boxes = _gather_feat(pred_boxes, ind)
        valid_mask = mask.bool()
        if valid_mask.sum() == 0:
            return iou_pred.new_tensor(0.0)

        target_boxes = gt_boxes[..., :7]
        pred_selected = pred_boxes[valid_mask]
        target_selected = target_boxes[valid_mask]
        iou_targets = aligned_boxes_iou3d_gpu(pred_selected.float(), target_selected.float()).view(-1)
        finite_mask = torch.isfinite(iou_targets)
        if finite_mask.sum() == 0:
            return iou_pred.new_tensor(0.0)
        iou_targets = iou_targets[finite_mask]
        pred_selected_iou = pred_iou[valid_mask][finite_mask]
        iou_targets = 2 * iou_targets - 1
        return F.l1_loss(pred_selected_iou, iou_targets, reduction='mean')


class IouRegLossCenterNet(nn.Module):
    def __init__(self, iou_type='DIoU'):
        super().__init__()
        if iou_type == 'DIoU':
            self.bbox3d_iou_func = bbox3d_overlaps_diou
        elif iou_type == 'GIoU':
            self.bbox3d_iou_func = bbox3d_overlaps_giou
        else:
            raise NotImplementedError(f'Unsupported IoU regression type: {iou_type}')

    def forward(self, pred_boxes, mask, ind, gt_boxes):
        valid_mask = mask.bool()
        if valid_mask.sum() == 0:
            return pred_boxes.new_tensor(0.0)

        pred_boxes = _gather_feat(pred_boxes, ind)
        iou = self.bbox3d_iou_func(pred_boxes[valid_mask], gt_boxes[valid_mask][:, :7])
        return (1.0 - iou).sum() / (valid_mask.sum() + 1e-4)


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        for name, head_cfg in sep_head_dict.items():
            output_channels = head_cfg['out_channels']
            num_conv = head_cfg['num_conv']
            layers = []
            for _ in range(num_conv - 1):
                layers.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU(),
                ))
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            head = nn.Sequential(*layers)
            if 'hm' in name:
                head[-1].bias.data.fill_(init_bias)
            else:
                for module in head.modules():
                    if isinstance(module, nn.Conv2d):
                        kaiming_normal_(module.weight.data)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
            setattr(self, name, head)

    def forward(self, features):
        return {name: getattr(self, name)(features) for name in self.sep_head_dict}


class RadarCenterHead(nn.Module):
    def __init__(
        self,
        model_cfg,
        input_channels,
        class_names,
        point_cloud_range,
        voxel_size,
        feature_key='radar_spatial_features_2d',
        pred_dicts_key='radar_pred_dicts',
        target_dicts_key='target_dicts',
        final_box_dict_key='final_box_dict',
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.class_names = class_names
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_key = feature_key
        self.pred_dicts_key = pred_dicts_key
        self.target_dicts_key = target_dicts_key
        self.final_box_dict_key = final_box_dict_key
        self.feature_map_stride = model_cfg['target_assigner_config']['feature_map_stride']
        self.predict_boxes_when_training = model_cfg.get('predict_boxes_when_training', True)
        self.rectifier = model_cfg.get('rectifier', 0.0)
        self.use_bias_before_norm = model_cfg.get('use_bias_before_norm', False)
        self.with_iou = 'iou' in model_cfg['separate_head_cfg']['head_dict']
        self.with_iou_reg = model_cfg.get('iou_reg', False)

        self.class_names_each_head = model_cfg['class_names_each_head']
        self.class_id_mapping_each_head = []
        total_classes = 0
        filtered_class_names_each_head = []
        for head_classes in self.class_names_each_head:
            filtered_head_classes = [name for name in head_classes if name in self.class_names]
            filtered_class_names_each_head.append(filtered_head_classes)
            total_classes += len(filtered_head_classes)
            mapping = [self.class_names.index(name) for name in filtered_head_classes]
            self.class_id_mapping_each_head.append(torch.tensor(mapping, dtype=torch.long))
        self.class_names_each_head = filtered_class_names_each_head
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                model_cfg['shared_conv_channel'],
                3,
                stride=1,
                padding=1,
                bias=self.use_bias_before_norm,
            ),
            nn.BatchNorm2d(model_cfg['shared_conv_channel']),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = model_cfg['separate_head_cfg']
        for head_classes in self.class_names_each_head:
            cur_head_dict = copy.deepcopy(self.separate_head_cfg['head_dict'])
            cur_head_dict['hm'] = {'out_channels': len(head_classes), 'num_conv': model_cfg['num_hm_conv']}
            self.heads_list.append(
                SeparateHead(
                    input_channels=model_cfg['shared_conv_channel'],
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.use_bias_before_norm,
                )
            )

        self.hm_loss_func = FocalLossCenterNet()
        self.reg_loss_func = RegLossCenterNet()
        self.iou_loss_func = IouLossCenterNet() if self.with_iou else None
        self.iou_reg_loss_func = IouRegLossCenterNet(self.with_iou_reg) if self.with_iou_reg else None
        self.forward_ret_dict = {}

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size):
        target_cfg = self.model_cfg['target_assigner_config']
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        reg_dim = sum(self.separate_head_cfg['head_dict'][name]['out_channels'] for name in self.separate_head_cfg['head_order'] if name != 'iou')
        ret_boxes = gt_boxes.new_zeros((target_cfg['num_max_objs'], reg_dim))
        gt_box = gt_boxes.new_zeros((target_cfg['num_max_objs'], 7))
        inds = gt_boxes.new_zeros(target_cfg['num_max_objs']).long()
        mask = gt_boxes.new_zeros(target_cfg['num_max_objs']).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx = gt_boxes[:, 3] / self.voxel_size[0] / self.feature_map_stride
        dy = gt_boxes[:, 4] / self.voxel_size[1] / self.feature_map_stride
        radius = torch.clamp_min(
            gaussian_radius(dx, dy, min_overlap=target_cfg['gaussian_overlap']).int(),
            min=target_cfg['min_radius'],
        )

        for index in range(min(target_cfg['num_max_objs'], gt_boxes.shape[0])):
            if dx[index] <= 0 or dy[index] <= 0:
                continue
            if not (0 <= center_int[index][0] <= feature_map_size[0] and 0 <= center_int[index][1] <= feature_map_size[1]):
                continue

            class_id = int(gt_boxes[index, -1].item() - 1)
            draw_gaussian_to_heatmap(heatmap[class_id], center[index], radius[index].item())
            inds[index] = center_int[index, 1] * feature_map_size[0] + center_int[index, 0]
            mask[index] = 1
            ret_boxes[index, 0:2] = center[index] - center_int_float[index]
            ret_boxes[index, 2] = z[index]
            ret_boxes[index, 3:6] = gt_boxes[index, 3:6].log()
            ret_boxes[index, 6] = torch.cos(gt_boxes[index, 6])
            ret_boxes[index, 7] = torch.sin(gt_boxes[index, 6])
            if reg_dim >= 10 and gt_boxes.shape[1] > 8:
                ret_boxes[index, 8:10] = gt_boxes[index, 7:9]
            gt_box[index, :7] = gt_boxes[index, :7]

        return heatmap, ret_boxes, inds, mask, gt_box

    def assign_targets(self, gt_boxes, feature_map_size):
        feature_map_size = feature_map_size[::-1]
        ret_dict = {'heatmaps': [], 'target_boxes': [], 'inds': [], 'masks': [], 'gt_box': []}
        all_names = np.array(['bg', *self.class_names])
        for head_index, head_classes in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_box_list = [], [], [], [], []
            for batch_index in range(gt_boxes.shape[0]):
                cur_gt_boxes = gt_boxes[batch_index]
                gt_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
                single_head_boxes = []
                for gt_index, name in enumerate(gt_names):
                    if name not in head_classes:
                        continue
                    temp_box = cur_gt_boxes[gt_index].clone()
                    temp_box[-1] = head_classes.index(name) + 1
                    single_head_boxes.append(temp_box[None, :])

                single_head_boxes = cur_gt_boxes[:0, :] if len(single_head_boxes) == 0 else torch.cat(single_head_boxes, dim=0)
                heatmap, ret_boxes, inds, mask, gt_box = self.assign_target_of_single_head(
                    num_classes=len(head_classes),
                    gt_boxes=single_head_boxes.cpu(),
                    feature_map_size=feature_map_size,
                )
                heatmap_list.append(heatmap.to(cur_gt_boxes.device))
                target_boxes_list.append(ret_boxes.to(cur_gt_boxes.device))
                inds_list.append(inds.to(cur_gt_boxes.device))
                masks_list.append(mask.to(cur_gt_boxes.device))
                gt_box_list.append(gt_box.to(cur_gt_boxes.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_box'].append(torch.stack(gt_box_list, dim=0))
        return ret_dict

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        code_weights = pred_dicts[0]['center'].new_tensor(self.model_cfg['loss_config']['loss_weights']['code_weights'])

        total_loss = 0
        tb_dict = {}
        for index, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = clip_sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][index])
            hm_loss = hm_loss * self.model_cfg['loss_config']['loss_weights']['cls_weight']

            pred_boxes = torch.cat([pred_dict[name] for name in self.separate_head_cfg['head_order'] if name != 'iou'], dim=1)
            reg_loss = self.reg_loss_func(pred_boxes, target_dicts['masks'][index], target_dicts['inds'][index], target_dicts['target_boxes'][index])
            loc_loss = (reg_loss * code_weights).sum() * self.model_cfg['loss_config']['loss_weights']['loc_weight']

            total_loss += hm_loss + loc_loss
            tb_dict[f'hm_loss_head_{index}'] = hm_loss.item()
            tb_dict[f'loc_loss_head_{index}'] = loc_loss.item()

            if self.iou_loss_func is not None or self.iou_reg_loss_func is not None:
                dense_boxes = self._build_dense_box_predictions(pred_dict)
                if self.iou_loss_func is not None and 'iou' in pred_dict:
                    iou_loss = self.iou_loss_func(
                        pred_dict['iou'],
                        target_dicts['masks'][index],
                        target_dicts['inds'][index],
                        dense_boxes.detach(),
                        target_dicts['gt_box'][index],
                    )
                    iou_weight = self.model_cfg['loss_config']['loss_weights'].get('iou_weight', 1.0)
                    total_loss += iou_weight * iou_loss
                    tb_dict[f'iou_loss_head_{index}'] = iou_loss.item()

                if self.iou_reg_loss_func is not None:
                    iou_reg_loss = self.iou_reg_loss_func(
                        dense_boxes,
                        target_dicts['masks'][index],
                        target_dicts['inds'][index],
                        target_dicts['gt_box'][index],
                    )
                    loc_weight = self.model_cfg['loss_config']['loss_weights'].get('loc_weight', 1.0)
                    total_loss += loc_weight * iou_reg_loss
                    tb_dict[f'iou_reg_loss_head_{index}'] = iou_reg_loss.item()

        tb_dict['rpn_loss'] = total_loss.item()
        return total_loss, tb_dict

    def _build_dense_box_predictions(self, pred_dict):
        batch_center = pred_dict['center'].permute(0, 2, 3, 1).contiguous()
        batch_center_z = pred_dict['center_z'].permute(0, 2, 3, 1).contiguous()
        batch_dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5)).permute(0, 2, 3, 1).contiguous()
        batch_rot = pred_dict['rot'].permute(0, 2, 3, 1).contiguous()
        batch_rot = torch.atan2(batch_rot[..., 1:2], batch_rot[..., 0:1])

        batch, height, width, _ = batch_dim.size()
        ys, xs = torch.meshgrid(
            torch.arange(0, height, device=batch_dim.device),
            torch.arange(0, width, device=batch_dim.device),
            indexing='ij',
        )
        ys = ys.view(1, height, width, 1).repeat(batch, 1, 1, 1)
        xs = xs.view(1, height, width, 1).repeat(batch, 1, 1, 1)

        xs = (xs + batch_center[..., 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (ys + batch_center[..., 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dense_boxes = torch.cat([xs, ys, batch_center_z, batch_dim, batch_rot], dim=-1)
        return dense_boxes.view(batch, -1, dense_boxes.shape[-1])

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_cfg = self.model_cfg['post_processing']
        post_center_limit_range = torch.tensor(post_cfg['post_center_limit_range'], device=pred_dicts[0]['hm'].device).float()
        results = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for _ in range(batch_size)]

        for head_index, pred_dict in enumerate(pred_dicts):
            class_mapping = self.class_id_mapping_each_head[head_index].to(pred_dict['hm'].device)
            batch_iou = None
            if 'iou' in pred_dict:
                batch_iou = (pred_dict['iou'] + 1.0) * 0.5
                batch_iou = batch_iou.type_as(pred_dict['dim'])

            final_pred_dicts = decode_bbox_from_heatmap(
                heatmap=pred_dict['hm'].sigmoid(),
                rot_cos=pred_dict['rot'][:, 0].unsqueeze(1),
                rot_sin=pred_dict['rot'][:, 1].unsqueeze(1),
                center=pred_dict['center'],
                center_z=pred_dict['center_z'],
                dim=pred_dict['dim'].exp(),
                vel=pred_dict['vel'] if 'vel' in self.separate_head_cfg['head_order'] else None,
                iou=batch_iou,
                rectifier=self.rectifier,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                topk=post_cfg['max_obj_per_sample'],
                score_thresh=post_cfg['score_thresh'],
                post_center_limit_range=post_center_limit_range,
            )

            for batch_index in range(batch_size):
                cur_pred_dict = final_pred_dicts[batch_index]
                cur_boxes = cur_pred_dict['pred_boxes']
                cur_scores = cur_pred_dict['pred_scores']
                cur_labels = cur_pred_dict['pred_labels'].long()
                if cur_boxes.shape[0] > 0:
                    selected, selected_scores = class_agnostic_nms(
                        box_scores=cur_scores,
                        box_preds=cur_boxes,
                        nms_config=post_cfg['nms_config'],
                        score_thresh=None,
                    )
                    cur_boxes = cur_boxes[selected]
                    cur_scores = selected_scores
                    cur_labels = cur_labels[selected]

                results[batch_index]['pred_boxes'].append(cur_boxes)
                results[batch_index]['pred_scores'].append(cur_scores)
                results[batch_index]['pred_labels'].append(class_mapping[cur_labels.long()] + 1 if cur_labels.numel() > 0 else cur_labels)

        for batch_index in range(batch_size):
            if results[batch_index]['pred_boxes']:
                results[batch_index]['pred_boxes'] = torch.cat(results[batch_index]['pred_boxes'], dim=0)
                results[batch_index]['pred_scores'] = torch.cat(results[batch_index]['pred_scores'], dim=0)
                results[batch_index]['pred_labels'] = torch.cat(results[batch_index]['pred_labels'], dim=0)
            else:
                device = pred_dicts[0]['hm'].device
                results[batch_index]['pred_boxes'] = torch.zeros((0, 7), device=device)
                results[batch_index]['pred_scores'] = torch.zeros((0,), device=device)
                results[batch_index]['pred_labels'] = torch.zeros((0,), dtype=torch.long, device=device)
        return results

    def forward(self, batch_dict):
        spatial_features_2d = batch_dict[self.feature_key]
        shared_features = self.shared_conv(spatial_features_2d)
        pred_dicts = [head(shared_features) for head in self.heads_list]
        batch_dict[self.pred_dicts_key] = pred_dicts

        if self.training or batch_dict.get('compute_loss', False):
            target_dicts = self.assign_targets(batch_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:])
            batch_dict[self.target_dicts_key] = target_dicts
            self.forward_ret_dict = {'pred_dicts': pred_dicts, 'target_dicts': target_dicts}

        if (not self.training) or self.predict_boxes_when_training:
            batch_dict[self.final_box_dict_key] = self.generate_predicted_boxes(batch_dict['batch_size'], pred_dicts)
        return batch_dict


class CenterHead(RadarCenterHead):
    def __init__(self, model_cfg, input_channels, class_names, point_cloud_range, voxel_size):
        super().__init__(
            model_cfg=model_cfg,
            input_channels=input_channels,
            class_names=class_names,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            feature_key='spatial_features_2d',
            pred_dicts_key='lidar_pred_dicts',
            target_dicts_key='lidar_target_dicts',
            final_box_dict_key='lidar_final_box_dict',
        )