    

import math
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.dynamic_assign import assign
from lib.losses.accuracy import accuracy
from lib.losses.lineiou_loss import liou_loss
from lib.losses.focal_loss import FocalLoss
import numpy as np
from imgaug.augmentables.lines import LineString, LineStringsOnImage

class CLR_Loss(nn.Module):
    def __init__(
            self,
            cfg
    ):
        super(CLR_Loss, self).__init__()
        self.cfg = cfg
        self.refine_layers = 3
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane
    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes
    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines
    def transform_annotation(self, old, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        # old_lanes = old
        old_lanes = old

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones(
            (self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
        lanes_endpoints = np.ones((self.max_lanes, 2))
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas)

            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        new_anno = lanes
        return new_anno
    def forward(self,
                output,
                batch,
                seg_loss, 
                cls_loss_weight=2.,
                xyt_loss_weight=0.5,
                iou_loss_weight=2.,
                seg_loss_weight=1.):
            
            cls_loss_weight = self.cfg.cls_loss_weight
            
            xyt_loss_weight = self.cfg.xyt_loss_weight
            
            iou_loss_weight = self.cfg.iou_loss_weight
            
            seg_loss_weight = self.cfg.seg_loss_weight

            predictions_lists = output
            # old_lanes = [
            # [[tuple(point) for point in object_points] for object_points in group]
            # for group in batch
            # ]
            # line_strings_org = self.lane_to_linestrings(old_lanes)
            # batch = self.linestrings_to_lanes(line_strings_org)
            targets = torch.tensor(np.asarray(batch)).clone().cuda()
            cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
            cls_loss = 0
            reg_xytl_loss = 0
            iou_loss = 0
            cls_acc = []
            # print(targets.size())
            cls_acc_stage = []
            for stage in range(self.refine_layers):
                predictions_list = predictions_lists[stage]
                for predictions, target in zip(predictions_list, targets):
                    # print(target[:, 1])
                    target = target[target[:, 1] == 1]
                    # print(len(target))
                    if len(target) == 0:
                        # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                        cls_target = predictions.new_zeros(predictions.shape[0]).long()
                        cls_pred = predictions[:, :2]
                        cls_loss = cls_loss + cls_criterion(
                            cls_pred, cls_target).sum()
                        continue

                    with torch.no_grad():
                        matched_row_inds, matched_col_inds = assign(
                            predictions, target, self.img_w, self.img_h)

                    # classification targets
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_target[matched_row_inds] = 1
                    cls_pred = predictions[:, :2]

                    # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                    reg_yxtl = predictions[matched_row_inds, 2:6]
                    reg_yxtl[:, 0] *= self.n_strips
                    reg_yxtl[:, 1] *= (self.img_w - 1)
                    reg_yxtl[:, 2] *= 180
                    reg_yxtl[:, 3] *= self.n_strips

                    target_yxtl = target[matched_col_inds, 2:6].clone()

                    # regression targets -> S coordinates (all transformed to absolute values)
                    reg_pred = predictions[matched_row_inds, 6:]
                    reg_pred *= (self.img_w - 1)
                    reg_targets = target[matched_col_inds, 6:].clone()

                    with torch.no_grad():
                        predictions_starts = torch.clamp(
                            (predictions[matched_row_inds, 2] *
                            self.n_strips).round().long(), 0,
                            self.n_strips)  # ensure the predictions starts is valid
                        target_starts = (target[matched_col_inds, 2] *
                                        self.n_strips).round().long()
                        target_yxtl[:, -1] -= (predictions_starts - target_starts
                                            )  # reg length

                    # Loss calculation
                    cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                    ) / target.shape[0]

                    target_yxtl[:, 0] *= self.n_strips
                    target_yxtl[:, 2] *= 180
                    reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                        reg_yxtl, target_yxtl,
                        reduction='none').mean()

                    iou_loss = iou_loss + liou_loss(
                        reg_pred, reg_targets,
                        self.img_w, length=15)

                    # calculate acc
                    cls_accuracy = accuracy(cls_pred, cls_target)
                    cls_acc_stage.append(cls_accuracy)
                # print(len(cls_acc_stage))
                # cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))

            # extra segmentation loss
            # seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),
            #                     batch['seg'].long())

            cls_loss /= (len(targets) * self.refine_layers)
            reg_xytl_loss /= (len(targets) * self.refine_layers)
            iou_loss /= (len(targets) * self.refine_layers)

            loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
                 + iou_loss * iou_loss_weight + seg_loss * seg_loss_weight

            return_value = {
                'loss': loss,
                'loss_stats': {
                    'loss': loss,
                    'cls_loss': cls_loss * cls_loss_weight,
                    'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
                    'seg_loss': seg_loss * seg_loss_weight,
                    'iou_loss': iou_loss * iou_loss_weight
                }
            }

            # for i in range(self.refine_layers):
            #     return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]

            return return_value['loss']
