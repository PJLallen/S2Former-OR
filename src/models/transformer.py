# ------------------------------------------------------------------------
# HOTR official code : src/models/transformer.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR & HOTR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .or_utils import MultiheadAttention
from .attention import MMG_teacher
from .feed_forward import MLP
from PIL import Image
import numpy as np
from torchvision.utils import save_image

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, multiview=False, pointfusion=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # encoder_layer_additional = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # encoder_norm_additional = nn.LayerNorm(d_model) if normalize_before else None
        # self.encoder_additional = TransformerEncoder(encoder_layer_additional, num_encoder_layers, encoder_norm_additional)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        if multiview:
            #  this part for multi view fusion
            multiviewFusion_layer = TransformerDecoderLayer_multiview(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            multiviewFusion_norm = nn.LayerNorm(d_model)
            #  2 is the number of fusion layers
            self.multiviewFusion = TransformerDecoder(multiviewFusion_layer, 2, multiviewFusion_norm,
                                              return_intermediate=False)

        if pointfusion:
            #  this part for points fusion
            self.points_mlp = MLP(291, 256, 256, 1)
            pointsFusion_layer = TransformerDecoderLayer_multiview(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            pointsFusion_norm = nn.LayerNorm(d_model)
            #  2 is the number of fusion layers
            self.pointsFusion = TransformerDecoder(pointsFusion_layer, 2, pointsFusion_norm,
                                              return_intermediate=False)

        # #  this part for points fusion
        # self.points_mlp = MLP(291, 256, 256, 1)
        # self.pointsFusion = MMG_teacher(dim_node=d_model, num_heads=nhead)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead



    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, src_multiview, mask_multiview, pos_embed_multiview, multiview_fusion=False, points_fusion=False, point_features=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        src_multiview_remain_shape = src_multiview.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        pos_embed_multiview_remain_shape = pos_embed_multiview.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        mask_multiview_remain_shape = mask_multiview.flatten(1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # ################### visualiazation ###################
        # vis = memory.permute(1, 2, 0).view(bs, c, h, w)[:, :1, :, :][0].permute(1, 2, 0).detach().cpu().numpy()
        # plt.imshow(vis)
        # plt.savefig(f"vis_before_multi.png")
        # ################### visualiazation ###################

        if multiview_fusion:
            memory_multiview_remain_shape = self.encoder(src_multiview_remain_shape,
                                                         src_key_padding_mask=mask_multiview_remain_shape,
                                                         pos=pos_embed_multiview_remain_shape)
            memory_multiview_split = memory_multiview_remain_shape.split(3, dim=1)
            memory_multiview = torch.cat([k.flatten(0, 1).unsqueeze(0) for k in memory_multiview_split], dim=0).permute(
                1, 0, 2)
            mask_multiview_split = mask_multiview_remain_shape.split(3, dim=0)
            mask_multiview = torch.cat([k.flatten(0, 1).unsqueeze(0) for k in mask_multiview_split], dim=0)
            pos_embed_multiview_split = pos_embed_multiview_remain_shape.split(3, dim=1)
            pos_embed_multiview = torch.cat([k.flatten(0, 1).unsqueeze(0) for k in pos_embed_multiview_split],
                                            dim=0).permute(1, 0, 2)
            memory = self.multiviewFusion(memory, memory_multiview, memory_key_padding_mask=mask_multiview,
                                          pos=pos_embed_multiview, query_pos=pos_embed)[0]

            # ################### visualiazation ###################
            # vis = memory.permute(1, 2, 0).view(bs, c, h, w)[:, :1, :, :][0].permute(1, 2, 0).detach().cpu().numpy()
            # # vis = (vis + 1) / 2.
            # plt.imshow(vis)
            # plt.savefig(f"vis_before_point.png")
            # ################### visualiazation ###################

        if points_fusion:
            point_features = self.points_mlp(point_features)
            memory = self.pointsFusion(memory, point_features.permute(1, 0, 2))[0]

            # ################### visualiazation ###################
            # vis = memory.permute(1, 2, 0).view(bs, c, h, w)[:, :1, :, :][0].permute(1, 2, 0).detach().cpu().numpy()
            # # vis = (vis + 1) / 2.
            # plt.imshow(vis)
            # plt.savefig(f"vis_after_point.png")
            # ################### visualiazation ###################

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        if multiview_fusion:
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), memory_multiview_remain_shape.permute(1, 2, 0).view(3*bs, c, h, w)
        else:
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), None

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_role_embedding: Optional[Tensor] = None,
                query_structure_encoding: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           memory_role_embedding=memory_role_embedding,
                           query_structure_encoding=query_structure_encoding)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, crossattn = False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.crossattn = crossattn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     memory_role_embedding: Optional[Tensor] = None,
                     query_structure_encoding: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos) ## !! key & value are different: key=pos+value
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              memory_role_embedding=query_structure_encoding)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   memory_role_embedding=memory_role_embedding)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_crossattn(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     memory_role_embedding: Optional[Tensor] = None,
                     query_structure_encoding: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(tgt, query_pos) ## !! key & value are different: key=pos+value
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask,
        #                       memory_role_embedding=query_structure_encoding)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   memory_role_embedding=memory_role_embedding)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    memory_role_embedding: Optional[Tensor] = None,
                    query_structure_encoding: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              memory_role_embedding=query_structure_encoding)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   memory_role_embedding=memory_role_embedding)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_role_embedding: Optional[Tensor] = None,
                query_structure_encoding: Optional[Tensor] = None):
        if self.crossattn:
            return self.forward_crossattn(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, memory_role_embedding, query_structure_encoding)
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, memory_role_embedding, query_structure_encoding)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, memory_role_embedding, query_structure_encoding)

class TransformerDecoderLayer_multiview(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_crossattn(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     memory_role_embedding: Optional[Tensor] = None,
                     query_structure_encoding: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(tgt, query_pos) ## !! key & value are different: key=pos+value
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask,
        #                       memory_role_embedding=query_structure_encoding)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   memory_role_embedding=memory_role_embedding)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_role_embedding: Optional[Tensor] = None,
                query_structure_encoding: Optional[Tensor] = None):

        return self.forward_crossattn(tgt, memory, tgt_mask, memory_mask,
                             tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, memory_role_embedding, query_structure_encoding)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        multiview=args.use_multiviewfusion,
        pointfusion=args.use_pointsfusion
    )


def build_hoi_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.hoi_nheads,
        dim_feedforward=args.hoi_dim_feedforward,
        num_encoder_layers=args.hoi_enc_layers,
        num_decoder_layers=args.hoi_dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
