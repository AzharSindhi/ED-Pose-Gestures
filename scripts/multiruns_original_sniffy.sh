#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=multirun_stratified
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --array=0-19  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_err.txt

# export SLURM_NPROCS=1

# module load python 
# module load cuda
#

# # python3 -m venv venv
# run bashrc
source ~/.bashrc
conda init bash
conda activate gestures_pose
# cd models/edpose/ops
# rm -rf build/
# rm -rf dist/
# rm -rf MultiScaleDeformableAttention.egg-info/
# python setup.py build install
# cd ../../..

# cd ../Deformable-DETR/models/ops
# sh ./make.sh
# python test.py

epoch=40
LR=0.0001
WEIGHT_DECAY=0.01
LR_DROP=30
NUM_GROUP=100
DN_NUMBER=100
N_QUERIES=900
N_CLASSES=1
# Create a run name with the combination of defined LR, weight_decay, num_group, etc.
commands=()
N=5

export EDPOSE_COCO_PATH=/net/cluster/azhar/datasets/gestuers_keypoints_cmac/smell-gesture-recognition/coco_directory_gestures

for ((i=0; i<N; i++))
do

    # # Create a run name with the combination of defined LR, weight_decay, num_group, etc.
    run_name="lr${LR}_wd${WEIGHT_DECAY}_lrd${LR_DROP}_ng${NUM_GROUP}_dn${DN_NUMBER}_orig"
    # Run the command with the random values and add it to the commands array
    command="python main.py  --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
        --output_dir logs/multiruns_sensorytest/edpose_finetune$i/all_coco/ \
        --options modelname=edpose num_classes=$N_CLASSES batch_size=2 epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco \
        --fix_size \
        --note $run_name
        --person_only"
    commands+=("$command")

done

# run every command
echo "==================================="
for i in "${!commands[@]}"; do
    echo "Running command $i: ${commands[$i]}"
    eval ${commands[$i]}
done