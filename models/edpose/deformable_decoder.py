import math, random
import torch.nn as nn
import copy
import os
from typing import Optional, List, Union
import warnings
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .transformer_deformable import DeformableTransformerDecoderLayer
from .utils import gen_encoder_output_proposals, sigmoid_focal_loss, MLP, _get_activation_fn, gen_sineembed_for_position
from .ops.modules.ms_deform_attn import MSDeformAttn
import pickle

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class DeformableDecoder(nn.Module):

    def __init__(self, 
                d_model=256, query_dim=4, nhead=8, 
                num_decoder_layers=6, 
                dim_feedforward=2048, dropout=0.0,
                activation="relu",
                modulate_hw_attn=False,
                # for deformable encoder
                deformable_decoder=False,
                num_feature_levels=1,
                dec_n_points=4,
                # evo of #anchors
                dec_layer_number=None,
                # for detach
                decoder_sa_type='sa', 
                module_seq=['sa', 'ca', 'ffn'],
                return_intermediate=True, 
                dec_layer_share=False,
                dec_pred_bbox_embed_share=False,
                dec_layer_dropout_prob=None,
                 ):
        super().__init__()

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points, 
                                                        decoder_sa_type=decoder_sa_type,
                                                        module_seq=module_seq)
        
        if num_decoder_layers > 0:
            self.layers = _get_clones(decoder_layer, num_decoder_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_decoder_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        # assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder
        self.bbox_embed = self.get_bbox_embed()
        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        # self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == self.num_layers
            # assert dec_layer_number[0] == 
            
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            raise NotImplementedError
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0
        # self.num_group=num_group
        self.rm_detach = None
        # self.num_dn = num_dn
        # self.hw = nn.Embedding(self.num_body_points,2)
        # self.seperate_token_for_class = seperate_token_for_class
        self._reset_parameters()
            
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_bbox_embed(self):
        _bbox_embed = MLP(self.d_model, self.d_model, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        if self.dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(self.num_layers)]
        return nn.ModuleList(box_embed_layerlist)
    

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None, #
                tgt_mask2: Optional[Tensor] = None, #
                memory_mask: Optional[Tensor] = None, #
                tgt_key_padding_mask: Optional[Tensor] = None, #
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None, # num_queries, bs, 2 with sigmoid applied
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                ):
        output = tgt
        intermediate = []
        # reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  
        for layer_id, layer in enumerate(self.layers):
            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2 
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points) # nq, bs, 256*2
                reference_points_input = None

            raw_query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                output = layer(
                    tgt = output,
                    tgt_query_pos = query_pos,
                    tgt_query_sine_embed = query_sine_embed,
                    tgt_key_padding_mask = tgt_key_padding_mask,
                    tgt_reference_points = reference_points_input,

                    memory = memory,
                    memory_key_padding_mask = memory_key_padding_mask,
                    memory_level_start_index = level_start_index,
                    memory_spatial_shapes = spatial_shapes,
                    memory_pos = pos,
                    self_attn_mask = tgt_mask,
                    cross_attn_mask = memory_mask
                )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

            reference_before_sigmoid = inverse_sigmoid(reference_points)
            delta_unsig = self.bbox_embed[layer_id](output)
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = outputs_unsig.sigmoid()
            
            if self.rm_detach and 'dec' in self.rm_detach:
                reference_points = new_reference_points
            else:
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                ref_points.append(new_reference_points)
        
        if self.return_intermediate:
            return [
                [itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
            ]
        else:
            return [self.norm(output).transpose(0, 1)], [new_reference_points.transpose(0, 1)]


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_deformable_decoder(args):
    return DeformableDecoder(
            d_model=args.hidden_dim, query_dim=args.query_dim, nhead=args.nheads, 
            num_decoder_layers=args.classifier_decoder_layers, 
            dim_feedforward=args.dim_feedforward, dropout=args.dropout,
            activation=args.transformer_activation,
            modulate_hw_attn=True,
            # for deformable encoder
            deformable_decoder=True,
            num_feature_levels=args.num_feature_levels,
            dec_n_points=args.dec_n_points,
            # evo of #anchors
            dec_layer_number=args.dec_layer_number,
            # for detach
            decoder_sa_type=args.decoder_sa_type, 
            module_seq=args.decoder_module_seq,
            return_intermediate=args.classifier_decoder_return_intermediate,
            dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share    
            )
