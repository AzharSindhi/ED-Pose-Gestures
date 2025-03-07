#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=classifiers_vcocoexps
#SBATCH --gres=gpu:a40:4
#SBATCH --array=0-5  # Adjust based on the number of experiments
#SBATCH --output=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/%x_%j_err.txt


WORK_DIR=/home/atuin/b193dc/b193dc14/mywork
#/net/cluster/azhar/datasets/SensoryGestureRecognition/data
# export PYTHONPATH=${venv}/bin/python 
cd $WORK_DIR/ED-Pose-Gestures/
#rm -rf slurm_logs/
## create environment
# echo "creating environment"

# # python3 -m venv venv
source models/edpose/ops/env/bin/activate
# cd models/edpose/ops
# rm -rf build/
# rm -rf dist/
# rm -rf MultiScaleDeformableAttention.egg-info/
# python setup.py build install
# cd ../../..

# cd ../Deformable-DETR/models/ops
# sh ./make.sh
# python test.py
# cd $WORK_DIR/ED-Pose-Gestures/

epoch=100
LR=0.0001
WEIGHT_DECAY=0.1
LR_DROP=30
NUM_GROUP=100
DN_NUMBER=100
N_QUERIES=900
BS=4
N_CLASSES=22
# Create a run name with the combination of defined LR, weight_decay, num_group, etc.
commands=()
N=2

readonly JOB_CLASS="vcoco"
readonly STAGING_DIR="/tmp/$USER-$JOB_CLASS"
(umask 0077; mkdir -p "$STAGING_DIR") || { echo "ERROR: creating $STAGING_DIR failed"; exit 1; }

(
  exec {FD}>"$STAGING_DIR/.lock"
  flock "$FD"

  # check if another job has staged data already
  if [ ! -f "$STAGING_DIR/.complete" ]; then
    # START OF STAGING

    # -------------------------------------------------------
    # TODO: place here the code to copy data to $STAGING_DIR
    # -------------------------------------------------------

    cp $WORK_DIR/datasets/vcoco_data_processed.zip $STAGING_DIR
    unzip -q $STAGING_DIR/vcoco_data_processed.zip -d $STAGING_DIR

    # -------------------------------------------------------

    # END OF STAGING 
    : > "$STAGING_DIR/.complete"
  fi
)

export EDPOSE_COCO_PATH=$STAGING_DIR/vcoco_data_processed

PORT=55234
N=1
for ((i=0; i<N; i++))
do
##### NO FINETUNE #####

    # add + i
    CURRENT_PORT=$((PORT + 2))
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_nofinetune"
    # Run the command with the random values and add it to the commands array
    command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
        --edpose_model_path logs/multiruns_vcoco_01_03/extratoken1/all_coco/checkpoint.pth \
        --output_dir logs/exps_extratokenclassifier_07_03_vcoco/full_vanilla_withextra_nofinetune$i/all_coco/ \
        --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=30 lr_drop=10 lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --find_unused_params \
        --seperate_classifier  \
        --classifier_type full \
        --seperate_token_for_class \
        --classifier_decoder_layers 6 \
        --note $run_name"
    
    commands+=("$command")

    CURRENT_PORT=$((PORT + 3))
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_nofinetune_clip"
    # Run the command with the random values and add it to the commands array
    command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
        --edpose_model_path logs/multiruns_vcoco_01_03/extratoken1/all_coco/checkpoint.pth \
        --output_dir logs/exps_extratokenclassifier_07_03_vcoco/full_vanilla_withextra_nofinetune_clip$i/all_coco/ \
        --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=30 lr_drop=10 lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --find_unused_params \
        --seperate_classifier  \
        --classifier_type full \
        --seperate_token_for_class \
        --use_clip_prior \
        --classifier_decoder_layers 6 \
        --note $run_name"
    
    commands+=("$command")




  ##### finetune ######

    CURRENT_PORT=$((PORT + 4))
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_finetune"
    # Run the command with the random values and add it to the commands array
    command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
        --edpose_model_path logs/multiruns_vcoco_01_03/extratoken1/all_coco/checkpoint.pth \
        --output_dir logs/exps_extratokenclassifier_07_03_vcoco/full_vanilla_withextra_finetune$i/all_coco/ \
        --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=50 lr_drop=15 lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --find_unused_params \
        --seperate_classifier  \
        --classifier_type full \
        --classifier_decoder_layers 6 \
        --seperate_token_for_class \
        --finetune_edpose \
        --note $run_name"
    
    commands+=("$command")


    # add + i
    CURRENT_PORT=$((PORT + 5))
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_finetune_clip"
    # Run the command with the random values and add it to the commands array
    command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
        --edpose_model_path logs/multiruns_vcoco_01_03/extratoken1/all_coco/checkpoint.pth \
        --output_dir logs/exps_extratokenclassifier_07_03_vcoco/full_vanilla_withextra_finetune_clip$i/all_coco/ \
        --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=50 lr_drop=15 lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --find_unused_params \
        --seperate_classifier  \
        --classifier_type full \
        --classifier_decoder_layers 6 \
        --seperate_token_for_class \
        --use_clip_prior \
        --finetune_edpose \
        --note $run_name"
    
    commands+=("$command")

    ##### original without previous weights ####
    CURRENT_PORT=$((PORT + 6))
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_original_finetune"
    # Run the command with the random values and add it to the commands array
    command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
        --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
        --output_dir logs/exps_extratokenclassifier_07_03_vcoco/full_vanilla_withextra_original_finetune$i/all_coco/ \
        --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --find_unused_params \
        --seperate_classifier  \
        --classifier_type full \
        --classifier_decoder_layers 6 \
        --seperate_token_for_class \
        --finetune_edpose \
        --note $run_name"
    
    commands+=("$command")

    CURRENT_PORT=$((PORT + 7))
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_original_clip"
    # Run the command with the random values and add it to the commands array
    command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
        --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
        --output_dir logs/exps_extratokenclassifier_07_03_vcoco/_full_vanilla_withextra_original_clip$i/all_coco/ \
        --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --find_unused_params \
        --seperate_classifier  \
        --classifier_type full \
        --classifier_decoder_layers 6 \
        --seperate_token_for_class \
        --use_clip_prior \
        --finetune_edpose \
        --note $run_name"
    
    commands+=("$command")

done

# logs/multiruns_vcoco_01_03/extratoken1/all_coco/checkpoint.pth
#--edpose_model_path logs/multiruns_vcoco_01_03/extratoken1/all_coco/checkpoint.pth
# submit the jobs to slurm
srun ${commands[$SLURM_ARRAY_TASK_ID]}
# eval ${commands[0]}
