import math

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from lib.utils.lane import Lane
from lib.losses.focal_loss import FocalLoss
from lib.losses.accuracy import accuracy
from lib.ops import nms

from lib.utils.roi_gather import ROIGather, LinearModule
# from lib.utils.seg_decoder import SegDecoder
from lib.utils.dynamic_assign import assign
from lib.losses.lineiou_loss import liou_loss


class CLRHead(nn.Module):
    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36):
        super(CLRHead, self).__init__()
        self.conf_threshold=0.3
        self.nms_thres=50
        self.nms_topk=13
        self.ori_img_h = 966
        self.ori_img_w = 1280
        self.max_lanes = 13
        self.img_w = 640
        self.img_h = 512
        self.cut_height = 0
        self.n_strips = 72 - 1
        self.n_offsets = 72
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim
        print(sample_points)
        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        # self.sample_x_indexs = (torch.linspace(
        #     0, 1, steps=self.sample_points, dtype=torch.float32) *
        #                         self.n_strips).long()
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        # self.prior_feat_ys = torch.flip(
        #     (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1])
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))
        # self.prior_ys = torch.linspace(1,
        #                                0,
        #                                steps=self.n_offsets,
        #                                dtype=torch.float32)
        self.prior_feat_channels = prior_feat_channels
        print('ntrip ne', self.n_strips)
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer(name='priors', tensor=init_priors)
        # self.priors = init_priors
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)
        # self.priors_on_featmap = priors_on_featmap

        # Add adapter layers to convert PaFPNELAN outputs to the expected channel size.
        # Note: Original PaFPNELAN returns a tuple like (C2, c5, c8, c12, c13, c16, c19),
        # and the last three features (c13, c16, c19) are used for refinement.
        # After reversing, the order becomes [c19, c16, c13] with channels [512, 256, 128].

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 + 1)  # n_offsets + length + start_x, start_y, theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)
        self.num_classes = 13 + 1
        weights = torch.ones(self.num_classes)
        self.bg_weight = 0.4
        weights[0] = self.bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=255,
                                          weight=weights)

        self.init_weights()
    def to_cuda(self):
        """Move all tensors in the class to GPU."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.cuda())  # Move tensor to GPU
    def init_weights(self):
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)


    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        '''

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        # computed_ys = torch.flip((1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1])

        # # Print values for debugging
        # print("prior_feat_ys:", self.prior_feat_ys)
        # print("computed_ys:", computed_ys)

        # # Check if they are equal
        # print("Are they equal?", torch.equal(self.prior_feat_ys, computed_ys))
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)
        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        if torch.isnan(prior_ys).any():
            print('co nannnnn')
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight  # (num_prop, 3)

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)

        priors[:, 2:5] = predictions.clone()
        # if priors.is_cuda:
        #     print('cuda nenenennen')
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

        # init priors on feature map
        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]

        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    # forward function here
    def forward(self, x, **kwargs):
        # x is expected to be a tuple of features from PaFPNELAN:
        # (C2, c5, c8, c12, c13, c16, c19)
        x = x[7:]
        batch_features = list(x[len(x) - self.refine_layers:])
        batch_size = batch_features[-1].shape[0]
        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()
        # print("Priors:", self.priors)
        if torch.isnan(self.priors).any():
            print("NaN found in priors.")

        priors = self.priors.repeat(batch_size, 1, 1)
        priors_on_featmap = self.priors_on_featmap.repeat(batch_size, 1, 1)

        predictions_lists = []
        prior_features_stages = []

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            if torch.isnan(prior_xs).any():
                print(f"NaN found in prior_xs at stage {stage}")

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)
            if torch.isnan(batch_prior_features).any():
                print(f"NaN found in batch_prior_features at stage {stage}")
            
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages,
                                        batch_features[stage], stage)
            fc_features = fc_features.view(num_priors, batch_size,
                                        -1).reshape(batch_size * num_priors,
                                                    self.fc_hidden_dim)
            if torch.isnan(fc_features).any():
                print(f"NaN found in fc_features at stage {stage}")

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()

            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            if torch.isnan(cls_features).any():
                print(f"NaN found in cls_features at stage {stage}")

            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)
            if torch.isnan(reg_features).any():
                print(f"NaN found in reg_features at stage {stage}")

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])
            reg = reg.reshape(batch_size, -1, reg.shape[1])

            if torch.isnan(cls_logits).any():
                print(f"NaN found in cls_logits at stage {stage}")
            if torch.isnan(reg).any():
                print(f"NaN found in reg at stage {stage}")

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits
            predictions[:, :, 2:5] += reg[:, :, :3]
            predictions[:, :, 5] = reg[:, :, 3]

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
            
            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                tran_tensor(predictions[..., 2])) * self.img_h /
                torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5)
            )) / (self.img_w - 1)

            if torch.isnan(predictions).any():
                print(f"NaN found in predictions at stage {stage}")

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]
            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if self.training:
            output = predictions_lists #{'predictions_lists': predictions_lists}
            return output
        return (self.get_lanes(predictions_lists[-1]), predictions_lists)

    def predictions_to_pred(self, predictions):
        '''
        Convert predictions to internal Lane structure for evaluation.
        '''
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (lane_ys * (self.ori_img_h - self.cut_height) +
                       self.cut_height) / self.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

  


    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        softmax = nn.Softmax(dim=1)

        decoded = []
        for predictions in output:
            # filter out the conf lower than conf threshold
            threshold = self.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            # print(scores)
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[...,
                            5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.nms_thres,
                top_k=self.max_lanes)
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded
