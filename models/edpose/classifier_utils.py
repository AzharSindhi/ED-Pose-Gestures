import torch

def dn_post_process2(outputs_class, pad_size):
    output_known_class = outputs_class[:, :pad_size, :]
    outputs_class = outputs_class[:, pad_size:, :]
    return outputs_class, output_known_class

def update_classification_information(out, outputs_class, aux_loss=True):
    dn_number = out["dn_number"]
    mask_dict_not_none = out["mask_dict_not_none"]
    if dn_number > 0 and mask_dict_not_none:
        outputs_class, dn_class_pred = dn_post_process2(outputs_class, out["num_tgt"])
        out.update(
                {
                    'dn_class_pred': dn_class_pred,
                }
            )   

    out['pred_logits'] = outputs_class
    # print(out['pred_logits'].shape[1], out['pred_boxes'].shape[1], out["pred_keypoints"].shape[1])
    assert out['pred_logits'].shape[1] == out['pred_boxes'].shape[1] == out["pred_keypoints"].shape[1]

    # if aux_loss:
    #     if dn_number > 0 and out["num_tgt"] > 0:
    #         out["aux_outputs"][-1].update({
    #             'dn_class_pred': dn_class_pred,
    #         })

    return out