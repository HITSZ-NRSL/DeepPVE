import time
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import MinkowskiEngine as ME

from learning.network.backbone.mk_backbones import MinkUNet14A


class HPRNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = MinkUNet14A(in_channels=3, out_channels=num_classes)

    def forward(self, end_points):
        # backbone
        points = end_points['points'].float()  # B, N, 3
        B, point_num, _ = points.shape
        # # inverse transform
        # feats_vector = end_points['feats'] / torch.norm(end_points['feats'], dim=-1, keepdim=True)
        # end_points['feats_inv_trans'] = 3 * feats_vector - end_points['feats']
        # sparse_input = ME.SparseTensor(end_points['feats_inv_trans'], end_points['coors'],
        #                                quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        sparse_input = ME.SparseTensor(end_points['feats'], end_points['coors'],
                                       quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        x = self.net(sparse_input).F  # B, num_classes, N', N' is the number of quantized coordinates
        # x = x[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)  # B, num_classes, N
        end_points['logits'] = x

        return end_points
