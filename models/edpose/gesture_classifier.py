import math
import torch
from torch import nn
from .deformable_decoder import build_deformable_decoder
from .edpose import build_edpose
from ..registry import MODULE_BUILD_FUNCS
from typing import List
import util.misc as utils
from .classifier_utils import update_classification_information
from copy import deepcopy

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
                use_deformable=False,
                classifier_type="full",
                box_detach_type="detach_clone",
                class_detach_type = "clone"):
        
        super(EdPoseClassifier, self).__init__()
        self.edpose_model = edpose_model
        self.edpose_weights_path = edpose_weights_path
        self.edpose_finetune_ignore = edpose_finetune_ignore
        self.edpose_num_box_layers = edpose_num_box_layers
        self.num_body_points = num_body_points
        self.use_deformable = use_deformable
        self.box_detach_type = box_detach_type
        self.class_detach_type = class_detach_type
        if edpose_weights_path:
            self.load_edpose_weights()
        if not finetune_edpose:
            self.edpose_model.eval()

        self.seperate_token_for_class = seperate_token_for_class
        self.classifier_type = classifier_type
        self.out_size = 100
        if self.classifier_type == "partial":
            self.set_size = 1 + self.seperate_token_for_class
        else:
            self.set_size = 18 + self.seperate_token_for_class
        
        self.ref_set_size = 1 + self.seperate_token_for_class + int(classifier_type == "full")

        self.apply_content_transform = True
        if self.set_size == 1:
            self.query_transform = nn.Identity()
            self.ref_transform = nn.Identity()
        else:
            self.query_transform = nn.Linear(self.set_size * 256, d_model)
            self.ref_transform = nn.Linear(4 * self.ref_set_size, 4)

        # decoder_layer = nn.TransformerDecoderLayer(d_model=self.set_size * d_model, nhead=8)
        # decoder = nn.TransformerDecoder(decoder_layer, 2)

        self.decoder = decoder
        self.finetune_edpose = finetune_edpose
        self.dn_number = dn_number
        _class_embed = nn.Linear(d_model, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        self.class_embed = _class_embed
        self.pos_embedding_proj = nn.Linear(4, d_model)
        self.class_projection = deepcopy(_class_embed)
        self.class_specific_queries = nn.Embedding(num_classes, d_model) # for learnable
        self.kpts_ref_proj = nn.Linear(self.num_body_points*3, 4)
        self.box_prior_weight = 0.3
        # classifer full

    def reshape_and_transform(self, input, target_dim = 256):
        """
        Transform the input vector to the required shape
        """
        b, seq_len, dim = input.shape
        input = input.view(b, self.out_size, -1)
        # inp = torch.reshape(inp, (b, self.out_size, -1))
        ## optional transform here
        if target_dim == 4:
            # transform bbox (reference vector) here from 18*4 into 4
            input = self.ref_transform(input)
        elif self.apply_content_transform:
            # transform the content(query vector) here from 18*256 into 256 or into anyother dimension
            input = self.query_transform(input)

        return input
    
    def box_return_detached(self, tensor):
        if self.box_detach_type == "detach":
            return tensor.detach()
        if self.box_detach_type == "clone":
            return tensor.clone()
        if self.box_detach_type == "detach_clone":
            return tensor.detach().clone()

        return tensor

    def class_return_detached(self, tensor):
        if self.class_detach_type == "detach":
            return tensor.detach()
        if self.class_detach_type == "clone":
            return tensor.clone()
        if self.class_detach_type == "detach_clone":
            return tensor.detach().clone()

        return tensor
    

    def inject_class_information(self, queries, class_logits):
        """
        Injects prior class information from box queries into decoder queries.

        Args:
            queries (torch.Tensor): The decoder queries of shape (b, num_queries, d_model).
            class_logits (torch.Tensor): The predicted class logits from box queries of shape (b, num_queries, num_classes).

        Returns:
            torch.Tensor: Updated decoder queries incorporating class information.
        """
        # Apply softmax to get confidence scores (class probabilities)
        pred_class_weights = torch.softmax(class_logits, dim=-1)  # Shape: (b, num_queries, num_classes)

        # Get the predicted class indices (argmax over class dimension)
        pred_classes = torch.argmax(pred_class_weights, dim=-1)  # Shape: (b, num_queries)

        # Get class embeddings from nn.Embedding layer
        pred_embeddings = self.class_specific_queries(pred_classes)  # Shape: (b, num_queries, d_model)

        # Ensure pred_class_weights matches embedding dimension
        class_tokens = self.class_specific_queries.weight  # (num_classes, d_model)

        # Compute confidence-weighted class embedding (proper shape handling)
        weighted_class_embeddings = torch.einsum("bqc,cd->bqd", pred_class_weights, class_tokens)  # (b, num_queries, d_model)

        # Inject into decoder queries
        return queries + self.box_prior_weight * weighted_class_embeddings
    # Shape: (b, num_queries, d_model)

    def extract_layer_output(self, edpose_out, idx):
        last_layer_all_queries = edpose_out["hs"][idx]
        # last_layer_all_reference = edpose_out["reference"][idx]
        self.dn_number = edpose_out["dn_number"]

        layer_hs_bbox_dn = self.box_return_detached(last_layer_all_queries[:,:self.dn_number,:])
        layer_ref_bbox_dn = None
        dn_class_logits = None
        if "dn_bbox_pred" in edpose_out:
            layer_ref_bbox_dn = self.box_return_detached(edpose_out["dn_bbox_pred"])#self.box_return_detached(last_layer_all_reference[:,:self.dn_number,:])
            dn_class_logits = self.class_projection(layer_hs_bbox_dn)
        
        last_layer_queries = last_layer_all_queries[:,self.dn_number:,:]
        # last_layer_references = last_layer_all_reference[:,self.dn_number:,:]
        input_queries_norm = self.box_return_detached(last_layer_queries[:,0::(self.num_body_points+1 + self.seperate_token_for_class),:])
        input_ref_norm = self.box_return_detached(edpose_out["pred_boxes"])#self.box_return_detached(last_layer_references[:,0::(self.num_body_points+1 + self.seperate_token_for_class),:])
        box_class_logits = self.class_projection(input_queries_norm)
        layer_hs_keypoint_norm = self.box_return_detached(last_layer_queries[:, self.edpose_model.kpt_index, :])
        layer_ref_keypoint_norm = self.box_return_detached(edpose_out["pred_keypoints"])

        if self.seperate_token_for_class:
            layer_hs_cls_norm = last_layer_queries[:,1::(self.num_body_points+1 + self.seperate_token_for_class),:]
            layer_ref_cls_norm = self.box_return_detached(edpose_out["pred_boxes"])

            input_queries_norm = torch.cat((input_queries_norm, layer_hs_cls_norm), dim=1)
            input_ref_norm = torch.cat((input_ref_norm, layer_ref_cls_norm), dim=1)

        if self.classifier_type == "full":
            input_queries_norm = torch.cat((input_queries_norm, layer_hs_keypoint_norm), dim=1)
            layer_kpts_ref = self.kpts_ref_proj(layer_ref_keypoint_norm)
            input_ref_norm = torch.cat((input_ref_norm, layer_kpts_ref), dim=1)
        
        # if self.classifier_type == "partial":
        #     layer_hs_keypoint_norm = torch.zeros_like()
        
        layer_hs_final_norm = self.reshape_and_transform(input_queries_norm)
        layer_ref_final_norm = self.reshape_and_transform(input_ref_norm, target_dim=4)
        
        return layer_hs_bbox_dn, dn_class_logits, layer_hs_final_norm, box_class_logits, layer_ref_bbox_dn, layer_ref_final_norm

    def forward_decoder(self, decoder_queries, refpoints_sigmoid, box_logits, edpose_out, return_last=True):
        if self.use_deformable:
            hs, ref = self.forward_deformable_decoder(decoder_queries, refpoints_sigmoid, edpose_out)
            if return_last:
                return hs[-1], ref[-1]
            else:
                return hs, ref
        else:
            return self.forward_vanilla_decoder(decoder_queries, refpoints_sigmoid, box_logits, edpose_out)
        
    def forward_deformable_decoder(self, decoder_queries, refpoints_sigmoid, edpose_out):
        hs, references = self.decoder(
            tgt=decoder_queries.transpose(0, 1), 
            memory=edpose_out["memory"].transpose(0, 1),#.clone(),
            memory_key_padding_mask=edpose_out["memory_key_padding_mask"],#.clone(), 
            pos=edpose_out["pos"].transpose(0, 1),#.clone(),
            reference_points=refpoints_sigmoid.transpose(0, 1),#.clone(),
            level_start_index=edpose_out["level_start_index"],#.clone(), 
            spatial_shapes=edpose_out["spatial_shapes"],#.clone(),
            valid_ratios=edpose_out["valid_ratios"],#.clone(),
            # tgt_mask=edpose_out["attn_mask"],
            # tgt_mask2=edpose_out["attn_mask2"]
        )
        return hs, references

    def forward_vanilla_decoder(self, decoder_queries, queries_ref, box_logits, edpose_out):
        # print(decoder_queries.transpose(0, 1).shape, edpose_out["memory"].transpose(0, 1).shape)
        decoder_queries = self.inject_class_information(decoder_queries, box_logits)
        memory = edpose_out["memory"].detach() if not self.finetune_edpose else edpose_out["memory"]
        pos_embeddings_transformed = self.pos_embedding_proj(queries_ref)
        decoder_queries = decoder_queries + pos_embeddings_transformed
        hs = self.decoder(
            tgt=decoder_queries.transpose(0, 1), 
            memory=memory.transpose(0, 1), 
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
        
        dn_queries, dn_logits, match_queries, match_logits, dn_ref, match_ref = self.extract_layer_output(edpose_out, -1)
        # layer_hs_bbox_dn, dn_class_logits, layer_hs_final_norm, box_class_logits, layer_ref_bbox_dn, layer_ref_final_norm

        if dn_ref is not None:
            decoder_queries = torch.cat((dn_queries, match_queries), dim=1)
            decoder_ref = torch.cat((dn_ref, match_ref), dim=1)
            box_logits = torch.cat((dn_logits, match_logits), dim=1)
            hs, _ = self.forward_decoder(decoder_queries, decoder_ref, box_logits, edpose_out)

        else:
            hs, _ = self.forward_decoder(match_queries, match_ref, match_logits, edpose_out)

        # # forward to the decoder
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
    
    edpose_model, criterion, postprocessors = build_edpose(args)
    if args.classifier_use_deformable:
        decoder = build_deformable_decoder(args)
    else:
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.hidden_dim, 
            nhead=args.nheads, 
            dim_feedforward=4 * args.hidden_dim,  # Standard setting for transformer models
            dropout=0.1,  # Explicit dropout for regularization
            activation="relu"
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.classifier_decoder_layers)
    
    model = EdPoseClassifier(edpose_model, decoder, args.num_classes, 
                            d_model=args.hidden_dim, 
                            dn_number=0 if args.no_dn else args.dn_number,
                            edpose_num_box_layers=args.num_box_decoder_layers,
                            finetune_edpose=args.finetune_edpose, 
                            seperate_token_for_class=args.seperate_token_for_class,
                            edpose_weights_path=args.edpose_model_path,
                            edpose_finetune_ignore=args.edpose_finetune_ignore,
                            cls_no_bias=args.cls_no_bias,
                            use_deformable=args.classifier_use_deformable,
                            classifier_type=args.classifier_type,
                            box_detach_type=args.box_detach_type,
                            class_detach_type=args.class_detach_type
                        )
    return model, criterion, postprocessors


if __name__ == "__main__":

    args_path = "args.pickle"
    with open(args_path, "rb") as f:
        args = pickle.load(f)
    
    model = build_classifier(args)
    print(model(None, None))