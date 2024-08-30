#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import math

import os
import json
from .backbones import ConvTransformerBackbone

#import adapool_cuda
#from adaPool import AdaPool1d

IDENT_FUNCT_DICT = {'psi_ava': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),
                    'grasp': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),
                    'Graspms': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),
                    'endovis_2018': lambda x,y: 'seq_{}_frame{:03d}.jpg'.format(x,y),
                    'endovis_2017': lambda x,y: 'seq{}_frame{:03d}.jpg'.format(x,y),
                    'cholec80': lambda x,y: 'video{:02d}/video{:02d}_{:06d}.png'.format(x,x,y),
                    'Cholec80ms': lambda x,y: 'video{:02d}/video{:02d}_{:06d}.png'.format(x,x,y),
                    'misaw': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),
                    'misawms': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),
                    'heichole': lambda x,y: 'video_{:02d}/{:05d}.png'.format(x,y),
                    'heicholems': lambda x,y: 'video_{:02d}/{:05d}.png'.format(x,y),
                    'phakir': lambda x,y: 'Video_{:02d}/{}/frame_{:06d}.png'.format(x,str((y//1000)*1000),y),
                    'Phakirms': lambda x,y: 'Video_{:02d}/{}/frame_{:06d}.png'.format(x,str(int((y//1000)*1000)),y),
                    }

FEATURE_SIZE = {'faster': 1024,
                'mask': 1024,
                'mask_max-mean': 1536,
                'mask_adaptive': 2048,
                'mask_all': 2560,
                'detr': 256,
                'm2f': 512}


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.projection_faster = nn.Linear(256, 1024, bias = True)
        
        self.projection_pathways = nn.Linear(sum(dim_in), 1024, bias = True)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(1024, num_classes, bias=True)
        self.act_func = act_func
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes, features=None):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection_pathways(x)
        
        if features is not None:
            features = features[:,1:]
            features = self.projection_faster(features)
        x = torch.cat((x, features), axis=1)

        x = self.projection(x)
        
        if self.training and self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        
        return x
        


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.extra_pool = nn.AvgPool3d([pool_size[0][0], pool_size[0][0], 1], stride=1)
        
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # handle extra heads need of extra pooling
        x = self.extra_pool(x)
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x



# Define your transformer-based model architecture
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


class TransformerBasicHead(nn.Module):
    """
    Frame Classification Head of TAPIS.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
        recognition=False
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.class_projection = nn.Linear(dim_in, num_classes, bias=True)
        self.cls_embed = cls_embed
        self.recognition = recognition
        self.act_func = act_func

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x, features=None, boxes_mask=None):
        if self.cls_embed and not self.recognition:
            x = x[:, 0]
        elif self.cls_embed:
            x = x[:,1:].mean(1)
        else:
            x = x.mean(1)
        breakpoint()
        if features:
            return x

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.class_projection(x)

        x = self.act(x)
        return x

class Contiguous(torch.nn.Module):
    def __init__(self):
        super(Contiguous, self).__init__()

    def forward(self,x):
        return x.contiguous()

class MultiSequenceTransformerBasicHead(nn.Module):
    """
    Frame Classification Head of TAPIS.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
        recognition=False,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(MultiSequenceTransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.dataset_name = cfg.TEST.DATASET
        self.parallel = cfg.NUM_GPUS > 1
        self.cls_embed = cls_embed
        self.recognition = recognition
        self.act_func = act_func
        self.multiscale_encoder = nn.ModuleList([])
        self.full_sequence_self_attention = nn.ModuleList([])
        self.num_classes = num_classes

        self.mvit_feats_enable = cfg.MVIT_FEATS.ENABLE
        self.mvit_feats_path = cfg.MVIT_FEATS.PATH

        self.full_self_attention = cfg.MVIT.FULL_SELF_ATTENTION
        self.full_self_attention_type = cfg.MVIT.FULL_SELF_ATTENTION_TYPE

        self.num_sequences = len(cfg.DATA.MULTI_SAMPLING_RATE)
        self.logit_join_type = cfg.MVIT.LOGIT_JOIN_TYPE

        self.cross_attention = cfg.MVIT.CROSS_ATTENTION 

        if self.cross_attention:
            for _ in range(self.num_sequences):
                self.multiscale_encoder.append(MultiScaleEncoder(depth=1,
                                                        sm_dim=768,
                                                        lg_dim=768,
                                                        cross_attn_depth = 2,
                                                        cross_attn_heads = 2,
                                                        cross_attn_dim_head = 64,
                                                        dropout=0.1
                                                        ))
            
            if self.full_self_attention:
                for _ in range(self.num_sequences):
                    self.self_attn_layers = nn.ModuleList([])
                    for i in range(2):
                        self.self_attn_layers.append(torch.nn.MultiheadAttention(embed_dim=768, 
                                                                                    num_heads=4, 
                                                                                    batch_first=True, 
                                                                                    dropout=0.1))
                    self.full_sequence_self_attention.append(self.self_attn_layers)
                    

        self.mlp_heads = nn.ModuleList([])

        if self.logit_join_type in ["sum", "ada"]:
            for _ in range(self.num_sequences):
                self.mlp_heads.append(nn.Sequential(nn.LayerNorm(768), nn.Linear(768, num_classes)))

        '''
        if self.logit_join_type in ["ada"]:
            self.pool = AdaPool1d(kernel_size=(self.num_sequences), beta=(1)).cuda()
            self.make_contiguous = Contiguous()
        '''

        if self.logit_join_type in ["mlp"]:
            input_size = 768 * self.num_sequences
            self.mlp_logits_embedding = nn.Sequential(
                                            nn.Dropout(p=0.1),
                                            nn.Linear(input_size, input_size, bias=True),
                                        )
            
            self.mlp_classifier = nn.Sequential(
                                            nn.Tanh(),
                                            nn.Dropout(p=0.1),
                                            nn.Linear(input_size, num_classes, bias=True)
                                        )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def save_cls_tokens(self, x, image_names):
        #json_path = "/home/aperezr20/endovis/GraSP/TAPIS/association_30fps.json"
        #json_data = self.upload_json_file(json_path)
        json_data = {}
        if self.parallel:
            image_names = [IDENT_FUNCT_DICT[self.dataset_name](*name) for name in image_names]
        
        for idx, frame_name in enumerate(image_names):
            if frame_name in json_data:
                name = json_data[frame_name]
            else:
                name = frame_name
            
            mvit_feats_dictionary = []
            video_name = name.split('/')[0]
            #print(video_name)
            if not os.path.exists(os.path.join(self.mvit_feats_path, video_name)):
                os.makedirs(os.path.join(self.mvit_feats_path, video_name))
            
            subvideo_name = name.split('/')[1]

            if not os.path.exists(os.path.join(self.mvit_feats_path, video_name, subvideo_name)):
                os.makedirs(os.path.join(self.mvit_feats_path, video_name, subvideo_name))

            sequence_embeddings = x[idx].data.cpu().numpy()
            mvit_feats_dictionary.extend([sequence_embeddings])
            feat_name = name.split('.')[0] + '.pth'
            path = os.path.join(self.mvit_feats_path, feat_name)
            torch.save(mvit_feats_dictionary, path)

    def upload_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    

    def forward(self, x, features=None, boxes_mask=None, image_names=None):
        b_size, seq_dim, embed_dim = x[0].shape
        all_sequences = x

        sequences = []

        for idx in range(self.num_sequences):
            main_seq = all_sequences[idx]
            other_seqs = all_sequences[:idx] + all_sequences[idx+1:]
            sequences.append((main_seq, tuple(other_seqs)))

        #logits_debug = torch.zeros((b_size, self.num_classes)).cuda()
        logits = []
        tokens = torch.zeros((b_size, len(x), embed_dim)).cuda()
        
        for idx, (seq_tokens, context) in enumerate(sequences):
            if self.cross_attention:
                encoded_seq = self.multiscale_encoder[idx](seq_tokens, context)

                if self.full_self_attention and self.full_self_attention_type == "cross_output":
                    context = torch.cat(context, dim=1)
                    encoded_seq = torch.cat((encoded_seq, seq_tokens), dim=1)
                    for i in range(2):
                        encoded_seq = self.full_sequence_self_attention[idx][i](encoded_seq, encoded_seq, encoded_seq)[0]
                
                cls_token = encoded_seq[:, 0]

                tokens[:, idx, :] = cls_token
            
            else:
                cls_token = seq_tokens[:, 0]

                tokens[:, idx, :] = cls_token

            if self.logit_join_type in ["sum", "ada"]:
                logits.append(self.mlp_heads[idx](cls_token))

            elif self.logit_join_type in ["mlp"]:
                logits.append(cls_token)
        
        logits = torch.stack(logits).cuda()

        if self.logit_join_type == "sum":
            logits = torch.sum(logits, dim=0)

        '''
        if self.logit_join_type == "ada":
            logits = logits.permute(1, 2, 0)
            logits = self.make_contiguous(logits)
            logits = self.pool(logits).squeeze()
        '''
        
        if self.logit_join_type == "mlp":
            logits = logits.permute(1, 0, 2)
            logits = logits.reshape(logits.shape[0], -1)
            embeddings = self.mlp_logits_embedding(logits)
            logits = self.mlp_classifier(embeddings)
        
        if self.act_func == "sigmoid" or not self.training:
            x = self.act(logits)

        if image_names is not None:
            self.save_cls_tokens(embeddings, image_names)

        return logits

class TransformerRoIHead(nn.Module):
    """
    Region classification head in TAPIS. 
    """

    def __init__(
        self,
        cfg,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
        gamma_init_value=1e-4
    ):
        
        super(TransformerRoIHead, self).__init__()
        self.cfg = cfg
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.cls_embed = cls_embed
        
        # Region features vector dimension 
        dim_features = cfg.FEATURES.DIM_FEATURES

        self.gamma = nn.Parameter(torch.ones(cfg.FEATURES.DIM_FEATURES) * gamma_init_value)

        # Use additional linear layers before temporal pooling
        self.use_prev = cfg.MODEL.TIME_MLP and cfg.MODEL.PREV_MLP
        
        if cfg.MODEL.DECODER:
            # Transform features to the same dimensions as MViT's output
            self.feat_adapter = nn.Sequential(nn.Linear(dim_features,dim_features,bias=True),
                                                nn.GELU(),
                                                nn.Linear(dim_features,dim_features, bias=True)
                                                )

            self.feat_project = nn.Sequential(nn.Linear(dim_features,768,bias=True),
                                                nn.GELU()
                                                )

            # Transformer decoder layer to do self-attention followed by cross-attention
            decoder_layer = nn.TransformerDecoderLayer(768, 
                                                       cfg.MODEL.DECODER_NUM_HEADS, 
                                                       dim_feedforward=cfg.MODEL.DECODER_HID_DIM,
                                                       batch_first=True)
            # Transformer decoder
            self.decoder = nn.TransformerDecoder(decoder_layer, 
                                                 cfg.MODEL.DECODER_NUM_LAYERS)
            dim_out = 768
            
        elif cfg.MODEL.TIME_MLP:
            if self.use_prev:
                # Linear layers previous to temporal pooling
                prev_layers = []
                for i in range(cfg.MODEL.PREV_MLP_LAYERS):
                    prev_layers.append(nn.Linear(cfg.MODEL.PREV_MLP_HID_DIM if i>0 else 768,
                                                cfg.MODEL.PREV_MLP_HID_DIM if i<cfg.MODEL.PREV_MLP_LAYERS-1 else cfg.MODEL.PREV_MLP_OUT_DIM,
                                                bias=True))
                    if i<cfg.MODEL.PREV_MLP_LAYERS-1:
                        prev_layers.append(nn.ReLU())
                self.prev_pool_project = nn.Sequential(*prev_layers)
            
            # Linear layers after temporal pooling
            post_layers = []
            for i in range(cfg.MODEL.POST_MLP_LAYERS):
                post_layers.append(nn.Linear(cfg.MODEL.POST_MLP_HID_DIM if i>0 else (cfg.MODEL.PREV_MLP_HID_DIM if self.use_prev else 768),
                                             cfg.MODEL.POST_MLP_HID_DIM if i<cfg.MODEL.POST_MLP_LAYERS-1 else cfg.MODEL.POST_MLP_OUT_DIM,
                                             bias=True))
                if i<cfg.MODEL.POST_MLP_LAYERS-1:
                    post_layers.append(nn.ReLU())
            self.post_pool_project = nn.Sequential(*post_layers)

            # Linear Layers to transform region feature vectors
            feat_layers = []
            for i in range(cfg.MODEL.FEAT_MLP_LAYERS):
                feat_layers.append(nn.Linear(cfg.MODEL.FEAT_MLP_HID_DIM if i>0 else dim_features,
                                             cfg.MODEL.FEAT_MLP_HID_DIM if i<cfg.MODEL.FEAT_MLP_LAYERS-1 else cfg.MODEL.FEAT_MLP_OUT_DIM,
                                             bias=True))
                if i<cfg.MODEL.FEAT_MLP_LAYERS-1:
                    feat_layers.append(nn.ReLU())
            self.feat_project = nn.Sequential(*feat_layers)
            
            dim_out = cfg.MODEL.FEAT_MLP_OUT_DIM + cfg.MODEL.POST_MLP_OUT_DIM
            
        else:
            self.mlp = nn.Sequential(nn.Linear(dim_features, 1024, bias=False),
                                    nn.BatchNorm1d(1024))
            dim_out = 1024 + 768
        
        # Final classification layer 
        self.class_projection = nn.Sequential(nn.Linear(dim_out, num_classes, bias=True),)
        
        self.act_func = act_func
        self.use_act = act_func == 'sigmoid' and cfg.TASKS.LOSS_FUNC[0] != 'bce_logit'
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
    
    def forward(self, inputs, features=None, boxes_mask=None):
        boxes_mask = boxes_mask.bool()

        if self.cls_embed:
            inputs = inputs[:, 1:, :]
        
        if self.cfg.MODEL.DECODER:
            features_after = self.feat_adapter(features)
            features = features + self.gamma * features_after
            features = self.feat_project(features)
            
            x = self.decoder(features, inputs, tgt_key_padding_mask=~boxes_mask)
            x = x[boxes_mask]
        
        else:
            if self.use_prev:
                inputs = self.prev_pool_project(inputs)
                
            x = inputs.mean(1)
            
            if self.cfg.MODEL.TIME_MLP:
                x = self.post_pool_project(x)

            max_boxes = boxes_mask.shape[-1] 
            
            # Repeat pooled time features to match the batch dimensions of box proposals
            x_boxes = x.unsqueeze(1).repeat(1,max_boxes,1)[boxes_mask] # Use box mask to remove padding
            
            features = features[boxes_mask] # Use box mask to remove padding
            features = self.feat_project(features)
            
            x = torch.cat([x_boxes, features], dim=1)

        x = self.class_projection(x)

        # Only apply final activation for validation or for bce loss
        if self.use_act or not self.training:
            x = self.act(x)

        return x


class ClassificationBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ClassificationBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        self.act_func = act_func

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        return x


class ClassificationRoIHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        cfg,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        max_len=1000
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ClassificationRoIHead, self).__init__()
        self.cfg = cfg

        self.num_frames = cfg.TEMPORAL_MODULE.NUM_FRAMES

        if dropout_rate > 0.0:
            self.dropout_1 = nn.Dropout(dropout_rate)

        self.act_func = act_func

        # Feature Vector dimension from deformable detr is 256
        self.dim_feats =  cfg.TEMPORAL_MODULE.INSFORMER_INPUT_DIM

        #----------------------------------------------
        # If we concatenate in feats
        self.embedding_1 = nn.Linear(self.dim_feats + cfg.TEMPORAL_MODULE.STEPFORMER_CAT_DIM, cfg.TEMPORAL_MODULE.INSFORMER_D_MODEL)
        #self.embedding_1 = nn.Linear(self.dim_feats, cfg.TEMPORAL_MODULE.INSFORMER_D_MODEL)

        self.positional_encoding_1 = PositionalEncoding(cfg.TEMPORAL_MODULE.INSFORMER_D_MODEL, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(cfg.TEMPORAL_MODULE.INSFORMER_D_MODEL, cfg.TEMPORAL_MODULE.INSFORMER_NUM_HEADS, batch_first=True)
        self.encoder_1 = nn.TransformerEncoder(encoder_layer, cfg.TEMPORAL_MODULE.INSFORMER_NUM_LAYERS)
    
        self.catfc_1 = nn.Linear(cfg.TEMPORAL_MODULE.INSFORMER_D_MODEL * cfg.TEMPORAL_MODULE.NUM_FRAMES, cfg.TEMPORAL_MODULE.INSFORMER_CAT_DIM)
        
        self.projection_1 = nn.Linear(cfg.TEMPORAL_MODULE.INSFORMER_CAT_DIM, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act_1 = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act_1 = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, features=None, boxes_mask=None, sequence_mask=None):
        x_t = inputs

        temporal_seq_len = x_t.shape[1]

        x_t = x_t.reshape((x_t.shape[0], -1))

        #max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self.training else self.cfg.DATA.MAX_BBOXES
        max_boxes = self.cfg.DATA.MAX_BBOXES 
        
        x_t = torch.unsqueeze(x_t, dim=1).repeat(1, max_boxes, 1)[boxes_mask]

        x_t = x_t.reshape((x_t.shape[0], temporal_seq_len, -1))

        if features is not None:

            features = features[boxes_mask] # Use box mask to remove padding
            sequence_mask = sequence_mask[boxes_mask]
            x = torch.reshape(features, (features.shape[0], self.num_frames, self.dim_feats))

            x = torch.cat((x, x_t), dim=2)
            
            x = self.embedding_1(x)
            x = self.positional_encoding_1(x)

            x = self.encoder_1(x, src_key_padding_mask=sequence_mask)

            x = x.reshape(x.shape[0], -1)
            x = self.catfc_1(x)

            if hasattr(self, "dropout"):
                x = self.dropout_1(x)

        elif features is None and self.cfg.FEATURES.ENABLE: # Hack for calculating model info
            x_boxes = torch.zeros(len(features), x.shape[1], device = inputs.device, requires_grad=True)
            features = torch.zeros(x_boxes.shape[0], self.dim_add, device = inputs.device, requires_grad=True)
            #x = torch.cat([x_boxes, features], dim=1)
            x = features
        else:
            raise NotImplementedError('There are no features')

        x = self.projection_1(x)

        # Only apply final activation for validation or for bce loss
        if self.training and self.act_func == "sigmoid":
            x = self.act_1(x)

        return x


class ActionFormerRoIHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        cfg,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        max_len=1000
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ActionFormerRoIHead, self).__init__()
        self.cfg = cfg

        self.num_frames = cfg.TEMPORAL_MODULE.NUM_FRAMES

        if dropout_rate > 0.0:
            self.dropout_1 = nn.Dropout(dropout_rate)

        self.act_func = act_func

        # Feature Vector dimension from deformable detr is 256
        self.dim_feats =  cfg.TEMPORAL_MODULE.INSFORMER_INPUT_DIM

        #----------------------------------------------

        self.catfc_1 = nn.Linear(cfg.TEMPORAL_MODULE.INSFORMER_D_MODEL * cfg.TEMPORAL_MODULE.NUM_FRAMES, cfg.TEMPORAL_MODULE.INSFORMER_CAT_DIM)
        
        self.projection_1 = nn.Linear(cfg.TEMPORAL_MODULE.INSFORMER_CAT_DIM, num_classes, bias=True)

        self.backbone = ConvTransformerBackbone(
                                n_in=self.dim_feats + cfg.TEMPORAL_MODULE.STEPFORMER_CAT_DIM,
                                n_embd=cfg.TEMPORAL_MODULE.INSFORMER_D_MODEL,
                                n_head=cfg.TEMPORAL_MODULE.INSFORMER_NUM_HEADS,
                                scale_factor=1
        )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act_1 = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act_1 = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, features=None, boxes_mask=None, sequence_mask=None):
        x_t = inputs
        
        temporal_seq_len = x_t.shape[1]

        x_t = x_t.reshape((x_t.shape[0], -1))
        
        #max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self.training else self.cfg.DATA.MAX_BBOXES
        max_boxes = self.cfg.DATA.MAX_BBOXES 
        
        x_t = torch.unsqueeze(x_t, dim=1).repeat(1, max_boxes, 1)[boxes_mask]

        x_t = x_t.reshape((x_t.shape[0], temporal_seq_len, -1))

        if features is not None:
            features = features[boxes_mask] # Use box mask to remove padding
            sequence_mask = sequence_mask[boxes_mask]
            
            x = torch.reshape(features, (features.shape[0], self.num_frames, self.dim_feats))
            
            sequence_mask = ~sequence_mask

            x = torch.cat((x, x_t), dim=2)

            x = x.cuda()

            x = torch.transpose(x, 1, 2)

            sequence_mask = torch.unsqueeze(sequence_mask,1)

            x, mask = self.backbone(x, sequence_mask)

            x = torch.stack(x, dim=0)
            
            x = torch.mean(x, dim=0)

            x = torch.transpose(x, 1, 2)

            x = x.reshape(x.shape[0], -1)

            x = self.catfc_1(x)

            if hasattr(self, "dropout"):
                x = self.dropout_1(x)

        elif features is None and self.cfg.FEATURES.ENABLE: # Hack for calculating model info
            x_boxes = torch.zeros(len(features), x.shape[1], device = inputs.device, requires_grad=True)
            features = torch.zeros(x_boxes.shape[0], self.dim_add, device = inputs.device, requires_grad=True)
            #x = torch.cat([x_boxes, features], dim=1)
            x = features
        else:
            raise NotImplementedError('There are no features')

        x = self.projection_1(x)

        # Only apply final activation for validation or for bce loss
        if self.training and self.act_func == "sigmoid":
            x = self.act_1(x)

        return x
