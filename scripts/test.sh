#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --job-name=vcoco_clone
#SBATCH --gres=gpu:a40:4
#SBATCH --partition=a40
#SBATCH --output=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/%x_%j_err.txt


WORK_DIR=/home/atuin/b193dc/b193dc14/mywork
#/net/cluster/azhar/datasets/SensoryGestureRecognition/data
# export PYTHONPATH=${venv}/bin/python 
cd $WORK_DIR/ED-Pose-Gestures/

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

echo "copying data"
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
PORT=33144

for ((i=0; i<N; i++))
do
    # add + i
    CURRENT_PORT=$((PORT + i))
  
    # # Create a run name with the combination of defined LR, weight_decay, num_group, etc.
    run_name="lr${LR}_wd${WEIGHT_DECAY}_lrd${LR_DROP}_ng${NUM_GROUP}_dn${DN_NUMBER}_orig"
    #    ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \

    command="torchrun --nproc_per_node=4 --master_port=$CURRENT_PORT main.py --config_file config/edpose.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
        --output_dir logs/test_runs_classifier_full_detclone_notran_cls/edpose_finetune$i/all_coco/ \
        --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --find_unused_params \
        --seperate_token_for_class \
        --seperate_classifier \
        --finetune_edpose \
        --note $run_name"
    commands+=("$command")

done

# submit the jobs to slurm
# srun ${commands[$SLURM_ARRAY_TASK_ID]}
eval ${commands[0]}
