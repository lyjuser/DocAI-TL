# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class SE_Module(nn.Module):
    def __init__(self, reduction_ratio=16, act_layer=nn.GELU):
        super().__init__()
        self.act = act_layer()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(512, 512 // reduction_ratio)
        self.fc2 = nn.Linear(512 // reduction_ratio, 512)
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        xs = self.maxpool(x)
        xs = self.fc1(xs)
        xs = self.act(xs)
        xs = torch.sigmoid(self.fc2(xs))
        y = x[:, :256, :, :] * xs[:, :256, :, :]
        z = x[:, 256:, :, :] * xs[:, 256:, :, :]
        x = y + z
        return x

def Feature_Recalibration_Module(fusion_features):
    B, C, H, W = fusion_features.shape # (B, 256, 64, 64)
    fusion_features_hat = fusion_features.flatten(2) # (B, 256, 4096)
    fusion_features_hat_t = fusion_features_hat.transpose(1, 2) # (B, 4096, 256)
    channel_attention_matrix = F.softmax(torch.matmul(fusion_features_hat, fusion_features_hat_t), dim=-1) # (B, 256, 256)
    weighted_fusion_features = torch.matmul(channel_attention_matrix, fusion_features_hat) # (B, 256, 4096)
    weighted_fusion_features = weighted_fusion_features.reshape(B, C, H, W)
    recalibration_features = fusion_features + weighted_fusion_features
    return recalibration_features

# Multiscale Adapter--------------------------------------------------------------------------------------------------------
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Multiscale_Adapter(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(Multiscale_Adapter, self).__init__()
        self.scale = scale
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, 256, kernel_size=1, stride=1, padding=0),
            BasicConv(256, 256, kernel_size=1, stride=1, padding=0),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, 256, kernel_size=1, stride=1, padding=0),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, 256, kernel_size=1, stride=1, padding=0),
            BasicConv(256, 256, kernel_size=5, stride=1, padding=2),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=5, dilation=5)
        )
        self.branch4 = BasicConv(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.convLinear = BasicConv(3 * 256, out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat((x1, x2, x3), 1)
        out = self.convLinear(out)
        out = x4 + self.scale * out
        out = self.relu(out)

        return out