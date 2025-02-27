#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=multirun_stratified
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --output=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/%x_%j_err.txt

WORK_DIR=/home/atuin/b193dc/b193dc14/mywork
cd $WORK_DIR/ED-Pose-Gestures/

# Activate environment
source models/edpose/ops/env/bin/activate

epoch=100
LR=0.0001
WEIGHT_DECAY=0.01
LR_DROP=30
NUM_GROUP=100
DN_NUMBER=100
N_QUERIES=900
BS=4
N_CLASSES=22
N=5  # Number of experiments per run

export EDPOSE_COCO_PATH=/home/atuin/b193dc/b193dc14/mywork/datasets/vcoco_data_processed

# Create job submission loop
for ((i=0; i<N; i++))
do
    export MASTER_PORT=$((12000 + i))  # Ensure a unique port per job
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_full_vanilla_noextra"

    command="sbatch --job-name=run_${i} --gres=gpu:a100:2 --partition=a100 --output=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/run_${i}_%j_out.txt --error=/home/atuin/b193dc/b193dc14/mywork/ED-Pose-Gestures/slurm_logs/run_${i}_%j_err.txt --wrap=\"torchrun --nproc_per_node=2 --master_port=$MASTER_PORT main.py --seperate_classifier --classifier_type full --config_file config/edpose.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. --output_dir logs/multiruns_vcoco_24_02/vanilla_full_noextra$i/all_coco/ --options modelname=classifier num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP --dataset_file=coco --find_unused_params --finetune_edpose --fix_size --note $run_name\""
    
    echo "Submitting job: $command"
    eval $command
done
