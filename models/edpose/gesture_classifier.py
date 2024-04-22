import math
import torch
from torch import nn
from .deformable_decoder import build_deformable_decoder
from .edpose import build_edpose
from ..registry import MODULE_BUILD_FUNCS
from typing import List
import util.misc as utils
from .classifier_utils import update_classification_information


import pickle

class EdPoseClassifier(nn.Module):
    def __init__(self, 
                edpose_model, decoder, num_classes, d_model = 256, dn_number = 100,
                num_body_points=17,
                edpose_num_box_layers = 2,
                finetune_edpose=False, seperate_token_for_class=False,
                edpose_weights_path=None,
                edpose_finetune_ignore=None,
                cls_no_bias=False,
                use_deformable=True):
        
        super(EdPoseClassifier, self).__init__()
        self.edpose_model = edpose_model
        self.edpose_weights_path = edpose_weights_path
        self.edpose_finetune_ignore = edpose_finetune_ignore
        self.edpose_num_box_layers = edpose_num_box_layers
        self.num_body_points = num_body_points
        self.use_deformable = use_deformable
        self.load_edpose_weights()
        # if not finetune_edpose:
        #     self.edpose_model.eval()

        self.decoder = decoder
        self.finetune_edpose = finetune_edpose
        self.seperate_token_for_class = seperate_token_for_class
        self.dn_number = dn_number
        _class_embed = nn.Linear(d_model, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        self.class_embed = _class_embed
    
    def extract_layer_output(self, edpose_out, idx):
        last_layer_all_queries = edpose_out["hs"][idx]
        last_layer_all_reference = edpose_out["reference"][idx]
        if self.seperate_token_for_class:
            slice_start = 1
        else:
            slice_start = 0
        
        layer_hs_bbox_dn = last_layer_all_queries[:,:self.dn_number,:]
        layer_hs_bbox_norm = last_layer_all_queries[:,self.dn_number:,:][:,slice_start::(self.num_body_points+1 + self.seperate_token_for_class),:]
        layer_ref_bbox_dn = last_layer_all_reference[:,:self.dn_number,:]
        layer_ref_bbox_norm = last_layer_all_reference[:,self.dn_number:,:][:,slice_start::(self.num_body_points+1 + self.seperate_token_for_class),:]
        return layer_hs_bbox_dn, layer_hs_bbox_norm, layer_ref_bbox_dn, layer_ref_bbox_norm

    def forward_decoder(self, decoder_queries, refpoints_sigmoid, edpose_out, return_last=True):
        if self.use_deformable:
            hs, ref = self.forward_deformable_decoder(decoder_queries, refpoints_sigmoid, edpose_out)
            if return_last:
                return hs[-1], ref[-1]
            else:
                return hs, ref
        else:
            return self.forward_vanilla_decoder(decoder_queries, edpose_out)
        
    def forward_deformable_decoder(self, decoder_queries, refpoints_sigmoid, edpose_out):
        hs, references = self.decoder(
            tgt=decoder_queries.transpose(0, 1), 
            memory=edpose_out["memory"].transpose(0, 1), 
            memory_key_padding_mask=edpose_out["memory_key_padding_mask"], 
            pos=edpose_out["pos"].transpose(0, 1),
            reference_points=refpoints_sigmoid.transpose(0, 1), 
            level_start_index=edpose_out["level_start_index"], 
            spatial_shapes=edpose_out["spatial_shapes"],
            valid_ratios=edpose_out["valid_ratios"],
            # tgt_mask=edpose_out["attn_mask"],
            # tgt_mask2=edpose_out["attn_mask2"]
        )
        return hs[-1], references[-1]

    def forward_vanilla_decoder(self, decoder_queries, edpose_out):
        hs = self.decoder(
            tgt=decoder_queries.transpose(0, 1), 
            memory=edpose_out["memory"].transpose(0, 1), 
            memory_key_padding_mask=edpose_out["memory_key_padding_mask"], 
            # pos=edpose_out["pos"].transpose(0, 1)
        )
        return hs.transpose(1, 0), None
    
    def forward(self, samples, targets:List=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not self.finetune_edpose:
            with torch.no_grad():
                edpose_out = self.edpose_model(samples, targets)
        else:
            edpose_out = self.edpose_model(samples, targets)
        
        dn_queries, match_queries, dn_ref, match_ref = self.extract_layer_output(edpose_out, -1) # last layer output
        decoder_queries = torch.cat((dn_queries, match_queries), dim=1)
        decoder_ref = torch.cat((dn_ref, match_ref), dim=1)
        # # forward to the decoder
        hs, _ = self.forward_decoder(decoder_queries, decoder_ref, edpose_out)
        if len(hs.shape) == 2: # batch size dimension missing when batch_size = 1
            hs = hs.unsqueeze(0)
        pred_class_logits = self.class_embed(hs)
        edpose_out = update_classification_information(edpose_out, pred_class_logits, aux_loss=False)
        return edpose_out


    def load_edpose_weights(self):
        checkpoint = torch.load(self.edpose_weights_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = self.edpose_finetune_ignore if self.edpose_finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        # logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = self.edpose_model.load_state_dict(_tmp_st, strict=False)

    # logger.info(str(_load_output))


@MODULE_BUILD_FUNCS.registe_with_name(module_name='classifier')
def build_classifier(args):
    args.classifier_decoder_layers = 2
    if args.classifier_use_deformable:
        decoder = build_deformable_decoder(args)
    else:
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden_dim, nhead=args.nheads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.classifier_decoder_layers)
    
    edpose_model, criterion, postprocessors = build_edpose(args)
    model = EdPoseClassifier(edpose_model, decoder, args.num_classes, 
                            d_model=args.hidden_dim, 
                            dn_number=0 if args.no_dn else args.dn_number,
                            edpose_num_box_layers=args.num_box_decoder_layers,
                            finetune_edpose=args.finetune_edpose, 
                            seperate_token_for_class=args.seperate_token_for_class,
                            edpose_weights_path=args.edpose_model_path,
                            edpose_finetune_ignore=args.edpose_finetune_ignore,
                            cls_no_bias=args.cls_no_bias,
                            use_deformable=args.classifier_use_deformable
                        )
    return model, criterion, postprocessors


if __name__ == "__main__":

    args_path = "args.pickle"
    with open(args_path, "rb") as f:
        args = pickle.load(f)
    
    model = build_classifier(args)
    print(model(None, None))