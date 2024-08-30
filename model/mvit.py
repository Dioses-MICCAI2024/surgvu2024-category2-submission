# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

from copy import deepcopy
import math
from functools import partial
import torch
import torch.nn as nn
import os
import json
import numpy as np

import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .attention import MultiScaleBlock
from .utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
)

from . import head_helper, stem_helper
#from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

_POOL1 = {
    "mvit": [[2, 1, 1]], 
    'MMViT': [[2, 1, 1]],
}

class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        
        # Prepare PSI-AVA tasks
        self.tasks = deepcopy(cfg.TASKS.TASKS)
        self.num_classes = deepcopy(cfg.TASKS.NUM_CLASSES)
        self.act_fun = deepcopy(cfg.TASKS.HEAD_ACT)
        self.regions = cfg.REGIONS.ENABLE
        self.recogn = cfg.TASKS.PRESENCE_RECOGNITION
        if cfg.REGIONS.ENABLE:
            self.features = cfg.FEATURES.ENABLE
            self._region_tasks = {task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS}
            self._frame_tasks = {task for task in cfg.TASKS.TASKS if task not in cfg.ENDOVIS_DATASET.REGION_TASKS}
            if cfg.TASKS.PRESENCE_RECOGNITION:
                self.recog_tasks = set(cfg.TASKS.PRESENCE_TASKS)
            
        # Prepare output.
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        # Following MocoV3, initializing with random patches stabilize optimization
        if cfg.MVIT.FREEZE_PATCH:
            self.patch_embed.requires_grad = False
            
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        self.embed_dim = dim_out
        self.norm = norm_layer(self.embed_dim)
        pool_size = _POOL1[cfg.MODEL.ARCH]
        pool_size[0][0] = self.patch_stride[0]

        self.mvit_feats_enable = cfg.MVIT_FEATS.ENABLE
        self.mvit_feats_path = cfg.MVIT_FEATS.PATH

        for idx, task in enumerate(self.tasks):
            if self.regions and task in self._region_tasks:
                if self.features:
                    extra_head = head_helper.TransformerRoIHead(
                                cfg,
                                num_classes=self.num_classes[idx],
                                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                                act_func=self.act_fun[idx],
                                cls_embed=self.cls_embed_on
                                )
                else:
                    pass

                if self.recogn and task in self.recog_tasks:
                    recog_head = head_helper.TransformerBasicHead(
                                    self.embed_dim,
                                    self.num_classes[idx],
                                    dropout_rate=cfg.MODEL.DROPOUT_RATE,
                                    act_func='sigmoid',
                                    cls_embed=False,
                                    recognition=True
                                )
                    self.add_module("extra_heads_{}_presence".format(task), recog_head)
            else:
                extra_head = head_helper.TransformerBasicHead(
                            self.embed_dim,
                            self.num_classes[idx],
                            dropout_rate=cfg.MODEL.DROPOUT_RATE,
                            act_func=self.act_fun[idx],
                            cls_embed=self.cls_embed_on,
                            recognition=False
                        )
            

            self.add_module("extra_heads_{}".format(task), extra_head)
   
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def upload_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def save_mvit_features(self, x, image_names):
        #image_names = [''.join(map(chr, lst)) for lst in image_names]

        json_path = "/media/SSD1/naparicioc/TAPIS_Transformer/TAPIS/association_30fps.json"
        json_data = self.upload_json_file(json_path)

        for idx, frame_name in enumerate(image_names):
            name = json_data[frame_name]
            mvit_feats_dictionary = []
            video_name = name.split('/')[0]
            #print(video_name)
            if not os.path.exists(os.path.join(self.mvit_feats_path, video_name)):
                os.mkdir(os.path.join(self.mvit_feats_path, video_name))

            full_feat = x[idx].data.cpu()
            cls_token = full_feat[0].numpy()
            mean_feats = torch.mean(full_feat[1:], axis=0).numpy()
            std_feats = torch.std(full_feat[1:], axis=0).numpy()
            min_feats = torch.min(full_feat[1:], axis=0).values.numpy()
            max_feats = torch.max(full_feat[1:], axis=0).values.numpy()
            mvit_feats_dictionary.extend([cls_token, mean_feats, max_feats])
            #mvit_feats_dictionary[name] = full_feat

            feat_name = name.split('.')[0] + '.pth'
            path = os.path.join(self.mvit_feats_path, feat_name)
            torch.save(mvit_feats_dictionary, path)


    def forward(self, x, features=None, boxes_mask=None, image_names=None):
        # breakpoint()
        out = {}
        if torch.cuda.is_available():
            x = x[0].cuda()
        else:
            x = x[0]  # Keep x on CPU if CUDA is not available
        
        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x) #Features

        # TAPIR head classification
        for task in self.tasks:
            extra_head = getattr(self, "extra_heads_{}".format(task))
            out[task] = extra_head(x, features, boxes_mask)

            if self.recogn and task in self.recog_tasks:
                out[f'{task}_presence'] = getattr(self, "extra_heads_{}_presence".format(task))(x, features, boxes_mask)
                
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=6000, max_batch_size=1):
        super().__init__()

        self.n_kv_heads = n_heads
        self.n_heads_q = n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x_q,
        x_k,
        x_v,
        mask,
    ):
        batch_size, seq_len, _ = x_q.shape  # (B, 1, Dim)

        xq = self.wq(x_q)
        xk = self.wk(x_k)
        xv = self.wv(x_v)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Repeat k, v for matching q dimensions
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores += mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, xv)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        output = self.wo(output)

        return output, scores # (B, 1, Dim) -> (B, 1, Dim)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, 256*4)
        
        # Layer Normalization
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
    def forward(self, enc_inputs, mask=None):
        ''' enc_inputs: [batch_size, src_len, d_model] '''
        
        # Self-Attention with Residual Connection
        attn_outputs, attn_scores = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, mask)
        attn_outputs = self.layernorm1(attn_outputs + enc_inputs)  # Residual Connection
        
        # Position-wise Feed-Forward with Residual Connection
        ffn_outputs = self.pos_ffn(attn_outputs)
        ffn_outputs = self.layernorm2(ffn_outputs + attn_outputs)  # Residual Connection
        
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        return ffn_outputs, attn_scores

class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.d_model = d_model
        self.n_heads = n_heads

    def make_zero_padding(self, x, current_length):
        target_length = 6000
        padding_length = target_length - current_length
        padding_idx = target_length - padding_length
        x = F.pad(x, (0, 0, 0, padding_length), "constant", 0)
        return x, padding_idx

    def forward(self, enc_inputs, mask=None, original_lenght=None):
        ''' enc_inputs: [batch_size, src_len, d_model] '''
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, attn = layer(enc_outputs, mask)
            enc_self_attns.append(attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, 256*4)

        # Layer Normalization
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, dec_inputs, enc_outputs, mask=None):
        ''' dec_inputs: [batch_size, tgt_len, d_model]
            enc_outputs: [batch_size, src_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
            dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # Masked Self-Attention with Residual Connection
        dec_outputs = dec_inputs
        
        dec_outputs, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, mask=mask)
        dec_outputs = self.layernorm1(dec_outputs + dec_inputs)  # Residual Connection

        # Encoder-Decoder Attention with Residual Connection
        dec_outputs1, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, mask=mask)
        dec_outputs = self.layernorm2(dec_outputs + dec_outputs1)  # Residual Connection

        # Position-wise Feed-Forward with Residual Connection
        dec_outputs2 = self.pos_ffn(dec_outputs)
        dec_outputs = self.layernorm3(dec_outputs + dec_outputs2)  # Residual Connection

        # [batch_size, tgt_len, d_model], [batch_size, h_heads, tgt_len, src_len]
        return dec_outputs, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, dec_inputs, enc_outputs, mask=None):
        ''' dec_inputs: [batch_size, tgt_len, d_model]
            enc_intpus: [batch_size, src_len, d_model]
            enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = dec_inputs

        dec_enc_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_enc_attn = layer(dec_outputs, enc_outputs, mask=mask)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_enc_attns

class VideoTransformerPerFrame(nn.Module):
    def __init__(self, cfg, classifier=True, max_len=15000):
        super(VideoTransformerPerFrame, self).__init__()
        self.tasks = cfg.TASKS.TASKS
        self.num_classes = cfg.TASKS.NUM_CLASSES
        self.act_fun = cfg.TASKS.HEAD_ACT
        self.seq_len = cfg.TEMPORAL_MODULE.NUM_FRAMES

        self.positional_encoding = PositionalEncoding(cfg.TEMPORAL_MODULE.STEPFORMER_D_MODEL, max_len)
        
        self.embedding = nn.Linear(cfg.TEMPORAL_MODULE.STEPFORMER_INPUT_DIM, cfg.TEMPORAL_MODULE.STEPFORMER_D_MODEL)

        self.encoder = Encoder(cfg.TEMPORAL_MODULE.STEPFORMER_D_MODEL, cfg.TEMPORAL_MODULE.STEPFORMER_NUM_LAYERS, cfg.TEMPORAL_MODULE.STEPFORMER_NUM_HEADS)
        self.decoder = Decoder(cfg.TEMPORAL_MODULE.STEPFORMER_D_MODEL, cfg.TEMPORAL_MODULE.STEPFORMER_NUM_LAYERS, cfg.TEMPORAL_MODULE.STEPFORMER_NUM_HEADS)
    
        self.classifier_encoder = nn.Linear(cfg.TEMPORAL_MODULE.STEPFORMER_D_MODEL, cfg.TEMPORAL_MODULE.STEPFORMER_D_MODEL)

        self.classifier = classifier
        
        if classifier:       
            for idx, task in enumerate(self.tasks):
                if task in ['actions', 'instruments']:
                    extra_head = head_helper.ClassificationRoIHead(
                            cfg, 
                            self.num_classes[idx],
                            dropout_rate=cfg.MODEL.DROPOUT_RATE,
                            act_func=self.act_fun[idx],
                            )
                
                else:
                    extra_head = head_helper.ClassificationBasicHead(
                            cfg,
                            cfg.TEMPORAL_MODULE.STEPFORMER_D_MODEL,
                            self.num_classes[idx],
                            dropout_rate=cfg.MODEL.DROPOUT_RATE,
                            act_func=self.act_fun[idx],
                            )

                self.add_module("extra_heads_{}".format(task), extra_head)
        
    def forward(self, x, features=None, boxes_mask=None, sequence_mask=None):
        out = {}

        x = x.cuda().float()
        
        x = self.embedding(x)
        x = self.positional_encoding(x)

        x, _ = self.encoder(x)
        x_tgt = self.classifier_encoder(x)

        x, _ = self.decoder(x, x_tgt)

        mid_frame = x.shape[1] // 2

        x = x[:, mid_frame] #-1 As is Online Setup. We are taking the current frame

        if self.classifier:
            for task in self.tasks:
                extra_head = getattr(self, "extra_heads_{}".format(task))
                if task == 'instruments' or task == 'actions':
                    sequence_mask = sequence_mask.reshape((sequence_mask.shape[0], sequence_mask.shape[1]//self.seq_len , self.seq_len))
                    out[task] = extra_head(x, features, boxes_mask, sequence_mask)
                else:
                    out[task] = extra_head(x)

        return out

