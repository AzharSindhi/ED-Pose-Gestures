#!/bin/bash

source models/edpose/ops/env/bin/activate
epoch=100
LR=0.0001
WEIGHT_DECAY=0.1
LR_DROP=30
NUM_GROUP=100
DN_NUMBER=100
N_QUERIES=900
BS=4
N_CLASSES=12

export EDPOSE_COCO_PATH=/home/atuin/b193dc/b193dc14/mywork/datasets/pascal_voc_actions_coco


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
    --output_dir logs/inference_pascalvoc_extratoken/all_coco/ \
    --options modelname=edpose num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
    set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
    --dataset_file=coco \
    --seperate_token_for_class \
    --no_distributed \
    --pretrain_model_path logs/pascal_04_03/extratoken1/all_coco/checkpoint.pth \
    --eval"

eval $command

