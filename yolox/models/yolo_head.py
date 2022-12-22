#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            # print('k:', k)
            # print('cls_feat:', cls_feat.shape)
            # print('cls_output:', cls_output.shape)
            # print('reg_feat:', reg_feat.shape)
            # print('reg_output:', reg_output.shape)
            # print('obj_output:', obj_output.shape)
            # print('=' * 10)
            # ==========
            # k: 0
            # cls_feat: torch.Size([1, 256, 40, 40])
            # cls_output: torch.Size([1, 80, 40, 40])
            # reg_feat: torch.Size([1, 256, 40, 40])
            # reg_output: torch.Size([1, 4, 40, 40])
            # obj_output: torch.Size([1, 1, 40, 40])
            # ==========
            # k: 1
            # cls_feat: torch.Size([1, 256, 20, 20])
            # cls_output: torch.Size([1, 80, 20, 20])
            # reg_feat: torch.Size([1, 256, 20, 20])
            # reg_output: torch.Size([1, 4, 20, 20])
            # obj_output: torch.Size([1, 1, 20, 20])
            # ==========
            # k: 2
            # cls_feat: torch.Size([1, 256, 10, 10])
            # cls_output: torch.Size([1, 80, 10, 10])
            # reg_feat: torch.Size([1, 256, 10, 10])
            # reg_output: torch.Size([1, 4, 10, 10])
            # obj_output: torch.Size([1, 1, 10, 10])
            # ==========
            # breakpoint()

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        # for output in outputs:
        #     print(output.shape)
        # 
        # torch.Size([2, 1600, 85])
        # torch.Size([2, 400, 85])
        # torch.Size([2, 100, 85])
        # NOTE: (B, H * W, 5 + #cls), where (H, W) is the shape of feature map.
        # Let A = sum of all H * W, 2100.
        # breakpoint()

        if self.training:
            return self.get_losses(
                # (B, 3, H, W), where (H, W) is the orignal image shape.
                imgs,
                # [ (1, H * W), ... ], x/y shifts for anchors in each feature map.
                x_shifts,
                y_shifts,
                # [ (1, H * W), ... ], downsample ratios (8, 16, 32) for anchors in each feature map.
                expanded_strides,
                # (B, L, 5), L ground-truth boxes (cls, x, y, w, h) ?(not sure) for each samples in batch.
                labels,
                # (B, A, 5 + #cls)
                torch.cat(outputs, 1),
                # Not empty if l1 is used.
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # (1, 1, H, W, 2)
            # grid[0][0][y][x] -> (x, y)
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        # n_anchors = 1.
        # (B, 5 + #cls, H, W) -> (B, 1, 5 + #cls, H, W)
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        # (B, 1, 5 + #cls, H, W)
        # -> (B, 1, H, W, 5 + #cls)
        # -> (B, H * W, 5 + #cls)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        # (1, 1, H, W, 2) -> (1, H * W, 2)
        grid = grid.view(1, -1, 2)
        # NOTE: Add grid (x, y) offsets.
        output[..., :2] = (output[..., :2] + grid) * stride
        # NOTE: Apply exp to (w, h)
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        # NOTE: output[..., 4] is IOU.
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        # NOTE: invalid labels are set to zeros.
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # (L, 4)
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                # (L,)
                gt_classes = labels[batch_idx, :num_gt, 0]
                # (A, 4)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                # breakpoint()

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # (MA*, #cls)
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                # (A, 1)
                obj_target = fg_mask.unsqueeze(-1)
                # (MA*, 4)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        # (SUM(MA* of each sample), #cls)
        cls_targets = torch.cat(cls_targets, 0)
        # (SUM(MA* of each sample), 4)
        reg_targets = torch.cat(reg_targets, 0)
        # (SUM(A * batch size), 1)
        obj_targets = torch.cat(obj_targets, 0)
        # (SUM(A * batch size),)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        # L
        num_gt,
        # A
        total_num_anchors,
        # (L, 4)
        gt_bboxes_per_image,
        # (L,)
        gt_classes,
        # (A, 4)
        bboxes_preds_per_image,
        # (1, A)
        expanded_strides,
        # (1, A)
        x_shifts,
        # (1, A)
        y_shifts,
        # (B, A, #cls)
        cls_preds,
        # (B, A, 4), not used.
        bbox_preds,
        # (B, A, 1)
        obj_preds,
        # (B, L, 5), not used.
        labels,
        # (B, 3, H, W)
        imgs,
        mode="gpu",
    ):
        # breakpoint()

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # (A,), (L, A*)
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        # (A*, 4)
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # (A*, #cls)
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        # (A*, 1)
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        # A*
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # (L, A*)
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        # (L, A*, #cls)
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        # Loss IOU.
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        # (L, A*)
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            # NOTE: (~is_in_boxes_and_center) for either in gt box or in agumented gt box, but not all.
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            # MA*
            num_fg,
            # (MA*,)
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        # NOTE: comment for debug.
        # if mode == "cpu":
        #     gt_matched_classes = gt_matched_classes.cuda()
        #     fg_mask = fg_mask.cuda()
        #     pred_ious_this_matching = pred_ious_this_matching.cuda()
        #     matched_gt_inds = matched_gt_inds.cuda()

        return (
            # (MA*,)
            gt_matched_classes,
            # (A,)
            fg_mask,
            # (MA*,)
            pred_ious_this_matching,
            # (MA*,)
            matched_gt_inds,
            # MA*
            num_fg,
        )

    def get_in_boxes_info(
        self,
        # (L, 4), (x, y, w, h)
        gt_bboxes_per_image,
        # (1, A)
        expanded_strides,
        # (1, A)
        x_shifts,
        # (1, A)
        y_shifts,
        # A
        total_num_anchors,
        # L
        num_gt,
    ):
        # breakpoint()

        # (A,)
        expanded_strides_per_image = expanded_strides[0]
        # NOTE: x/y shifts are defined in feature map scale.
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        # (1, A), the centers for each anchor.
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        # breakpoint()

        # NOTE: (x, y) is the center of box.
        # (L, A), expand each label up/down/left/right A times.
        gt_bboxes_per_image_l = (
            # (L,)
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            # (L, 1)
            .unsqueeze(1)
            # (L, A)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # NOTE: pretty expensive.
        # (L, A)
        # Calculate the delta between the center x of anchor and gt left & right.
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        # Calculate the delta between the center y of anchor and gt up & down.
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # (L, A, 4), deltas of left, right, up, down
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        # All positive -> the center of anchor in ground-truth labeling box.
        # (L, A), is_in_boxes[i][j] for the center of anchor j in labeling box i.
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        # NOTE:
        # (A,), is_in_boxes_all[j] for the center of anchor j in at least one labeling box.
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center
        # breakpoint()

        # NOTE: "Multi positives" in the paper?
        # And is NOT "center 3×3 area as positives"?
        center_radius = 2.5

        # (L, A), kind of use center_radius as scale to create bounding box.
        # The only difference is to use (center_radius * upsampling ratio) as height / width.
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        # (A,), is_in_centers_all[j] for the center of anchor j in at least one *generated* labeling box.
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        # NOTE: in labeling box OR in the center of labeling box.
        # (A,)
        # A* = sum(is_in_boxes_anchor)
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        # (L, A*), in labeling box AND in the center of labeling box.
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # (L, min(10, A*))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        # (L,), use int(sum of IOU) to define dynamic k.
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            # dynamic k samllest costs as positive samples.
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        # (A*,)
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            # At least one anchor match multiple ground-truths.
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            # Keep only the minimum ones.
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        # (A*,), flags for matched anchors.
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        # MA*
        num_fg = fg_mask_inboxes.sum().item()

        # Update valid anchors in-place, this changes A*.
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # Select mached gt classes.
        # (MA*,)
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        # (MA*,), selected matched IOU.
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
