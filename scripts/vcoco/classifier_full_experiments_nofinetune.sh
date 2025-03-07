#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=clexpsf_nfdef
#SBATCH --gres=gpu:a40:4
#SBATCH --partition=a40
#SBATCH --array=0-4  # Adjust based on the number of experiments
#SBATCH --output=slurm_logs/%x_%j_out.txt
#SBATCH --error=slurm_logs/%x_%j_err.txt


WORK_DIR=$WORK/mywork
#/net/cluster/azhar/datasets/SensoryGestureRecognition/data
# export PYTHONPATH=${venv}/bin/python 
cd $WORK_DIR/ED-Pose-Gestures/
#rm -rf slurm_logs/
## create environment
# echo "creating environment"

# # python3 -m venv venv
source models/edpose/ops/env/bin/activate


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

# add + i
CURRENT_PORT=$((PORT + 0))
run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_clip_transform"
# Run the command with the random values and add it to the commands array
command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
    --edpose_model_path logs/multiruns_vcoco_01_03/extratoken0/all_coco/checkpoint.pth \
    --output_dir logs/classifier_exps_06_03full_nofinedf/$run_name/all_coco \
    --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
    set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
    --dataset_file=coco \
    --fix_size \
    --find_unused_params \
    --seperate_classifier  \
    --classifier_type full \
    --classifier_decoder_layers 4 \
    --classifier_use_deformable \
    --seperate_token_for_class \
    --use_clip_prior \
    --queries_transform \
    --note $run_name"

commands+=("$command")

# add + i
CURRENT_PORT=$((PORT + 1))
run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_class_transform"
command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
    --edpose_model_path logs/multiruns_vcoco_01_03/extratoken0/all_coco/checkpoint.pth \
    --output_dir logs/classifier_exps_06_03full_nofinedf/$run_name/all_coco \
    --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
    set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
    --dataset_file=coco \
    --fix_size \
    --find_unused_params \
    --seperate_classifier  \
    --classifier_type full \
    --classifier_decoder_layers 4 \
    --classifier_use_deformable \
    --seperate_token_for_class \
    --use_class_prior \
    --queries_transform \
    --note $run_name"

commands+=("$command")

### NO TRANSFORM###
# add + i
CURRENT_PORT=$((PORT + 2))
run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_clip"
# Run the command with the random values and add it to the commands array
command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
    --edpose_model_path logs/multiruns_vcoco_01_03/extratoken0/all_coco/checkpoint.pth \
    --output_dir logs/classifier_exps_06_03full_nofinedf/$run_name/all_coco \
    --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
    set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
    --dataset_file=coco \
    --fix_size \
    --find_unused_params \
    --seperate_classifier  \
    --classifier_type full \
    --classifier_decoder_layers 4 \
    --classifier_use_deformable \
    --seperate_token_for_class \
    --use_clip_prior \
    --note $run_name"

commands+=("$command")

# add + i
CURRENT_PORT=$((PORT + 3))
run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra_class"
command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
    --edpose_model_path logs/multiruns_vcoco_01_03/extratoken0/all_coco/checkpoint.pth \
    --output_dir logs/classifier_exps_06_03full_nofinedf/$run_name/all_coco \
    --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
    set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
    --dataset_file=coco \
    --fix_size \
    --find_unused_params \
    --seperate_classifier  \
    --classifier_type full \
    --classifier_decoder_layers 4 \
    --classifier_use_deformable \
    --seperate_token_for_class \
    --use_class_prior \
    --note $run_name"

commands+=("$command")


#### WITHOUT ANYTHING #######
# add + i
CURRENT_PORT=$((PORT + 4))
run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_withextra"
# Run the command with the random values and add it to the commands array
command="torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py \
    --edpose_model_path logs/multiruns_vcoco_01_03/extratoken0/all_coco/checkpoint.pth \
    --output_dir logs/classifier_exps_06_03full_nofinedf/$run_name/all_coco \
    --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
    set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
    --dataset_file=coco \
    --fix_size \
    --find_unused_params \
    --seperate_classifier  \
    --classifier_type full \
    --classifier_use_deformable \
    --classifier_decoder_layers 4 \
    --seperate_token_for_class \
    --note $run_name"

commands+=("$command")


srun ${commands[$SLURM_ARRAY_TASK_ID]}
