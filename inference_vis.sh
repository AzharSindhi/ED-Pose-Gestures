#!/bin/bash

epoch=100
LR=0.0001
WEIGHT_DECAY=0.01
LR_DROP=30
NUM_GROUP=100
DN_NUMBER=100
N_QUERIES=900
BS=2
N_CLASSES=22

export EDPOSE_COCO_PATH=/net/cluster/azhar/mywork/coco_actions/v-coco/data/vcoco_data_processed


# command="python main.py  --seperate_classifier --classifier_type full --config_file config/edpose.cfg.py \
#     --output_dir logs/multiruns_vcoco_24_02/vanilla_full_noextra$i/all_coco/ \
#     --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
#     set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
#     --dataset_file=coco --find_unused_params \
#     --finetune_edpose \
#     --fix_size \
#     --pretrain_model_path "logs/multiruns_vcoco_23_02/vanilla_full_noextra0/all_coco/checkpoint.pth" \
#     --eval"

command="python main.py  --config_file config/edpose.cfg.py \
    --output_dir logs/multiruns_vcoco_24_02/edpose_finetune$i/all_coco/ \
    --options modelname=edpose num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
    set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
    --dataset_file=coco \
    --fix_size \
    --pretrain_model_path "logs/multiruns_vcoco_23_02/edpose_finetune0/all_coco/checkpoint.pth" \
    --eval"
eval $command

