import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *

import math
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from functools import partial

sys.path.append('/data2/POINT4D/UST-SSM/modules')
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .block import Block
from .CTS import *
from ipdb import set_trace as st
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder_1(nn.Module): 
    def __init__(self, encoder_channel,dim_in):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.dim_in = dim_in    
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.dim_in, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.dim_in)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0] 
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  
        feature = self.second_conv(feature) 
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] 
        return feature_global.reshape(bs, g, self.encoder_channel)


class Decoder_1(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))  
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  
        feature = self.second_conv(feature) 
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  
        return feature_global.reshape(bs, g, self.encoder_channel)


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class UST(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                               
                 temporal_kernel_size, temporal_stride,                                 
                 dim, depth, heads,                               
                 mlp_dim, num_classes, dropout, hos_branches_num, encoder_channel):                         
        super().__init__()
        self.hos_branches_num = hos_branches_num
        self.depth = depth

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, dim)
        )

        self.emb_relu = nn.ReLU()


        self.ssm_blocks = nn.ModuleList()
        initial_k_size = 12
        for i in range(self.hos_branches_num):
            branch_ssm_blocks = nn.ModuleList()
            for _ in range(self.depth):
                k_size = max(1, initial_k_size // (2 ** i))
                ssm_block = AggregationSSM(
                    dim=dim,
                    num_group=768,
                    num_heads=heads,
                    drop_path=0.1,
                    k_size=k_size,
                )
                branch_ssm_blocks.append(ssm_block)
            self.ssm_blocks.append(branch_ssm_blocks)

        self.encoder = Encoder_1(encoder_channel=encoder_channel,dim_in = dim)   # B L C*n        
        self.decoder = Decoder_1(encoder_channel=dim // 2)   # B L C*n   

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hos_branches_num * dim + dim // 2),
            nn.Linear(self.hos_branches_num * dim + dim // 2, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)   # [B, L, n, 3], [B, L, C, n]
        B, L, n, _ = xyzs.shape
        C = features.shape[2]

        features = features.permute(0, 1, 3, 2)
        xyzs, features = sort_point_clouds_hilbert(xyzs, features, num_dims=3, num_bits=8)
        features = features.permute(0, 1, 3, 2)

        xyzts = []
        xyzs_split = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs_split = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs_split]
        for t, xyz in enumerate(xyzs_split):
            t_val = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t + 1)
            t_val = torch.div(t_val, xyzs.shape[1], rounding_mode='floor')
            xyzt = torch.cat(tensors=(xyz, t_val), dim=2)
            xyzts.append(xyzt)

        xyzts = torch.stack(tensors=xyzts, dim=1)
        pos = self.pos_embed(xyzts)

        features = features.permute(0, 1, 3, 2)
        downsampling_factors = [2 ** i for i in range(self.hos_branches_num)]
        outputs = []
        main_x_output = None  

        for i, downsampling_factor in enumerate(downsampling_factors):
            if downsampling_factor == 1:
                indices = list(range(L))
            else:
                indices = [max(0, min(L - 1, k)) for k in range(0, L, downsampling_factor)]

            xyzs_branch = xyzs[:, indices, :, :]  # [B, W, n, 3]
            features_branch = features[:, indices, :, :]  # [B, W, C, n]
            W = xyzs_branch.shape[1]

            xyzs_branch = xyzs_branch.reshape(B, W * n, 3)
            pos_branch = pos[:, indices, :, :]
            pos_branch = pos_branch.reshape(B, W * n, pos_branch.shape[-1])

            features_branch = features_branch.reshape(B, W * n, C)
            x_branch = features_branch + pos_branch  # [B, W*n, C]

            x_output = x_branch
            for ssm_block in self.ssm_blocks[i]:
                x_output = ssm_block(center=xyzs_branch, x=x_output)  # [B, W*n, C_out]

            if downsampling_factor == 1:
                main_x_output = x_output  # [B, L*n, C_out]

            output_branch = torch.max(x_output, dim=1, keepdim=False)[0]  # [B, C_out]
            outputs.append(output_branch)


        feat = main_x_output.reshape(B, L, n, -1)  # [B, L, n, C_out]
        key = self.encoder(feat)
        key = key.reshape(key.shape[0], key.shape[1], key.shape[2] // 3, 3) 
        key1 = key
        key = self.decoder(key)   # [B, L, dim//2]
        key_output = torch.max(key, dim=1, keepdim=False)[0]  # [B, dim//2]


        outputs_concat = [key_output] + outputs  # [key_output, output0, output1, output2, ...]
        output = torch.cat(outputs_concat, dim=1)  # [B, dim//2 + hos_branches_num * C_out]

        output = self.mlp_head(output)  # [B, num_classes]

        return output, key1
