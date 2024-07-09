#!/bin/bash
#SBATCH --time=09:00:00
#SBATCH --job-name=sensoryArt
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --output=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_err.txt

# export SLURM_NPROCS=1
# --array=0-1  # Adjust based on the number of experiments

# module load python 
# module load cuda
WORK_DIR=/home/woody/iwi5/iwi5197h
export PYTHONPATH=${venv}/bin/python 
cd $WORK_DIR/ED-Pose-Gestures/
#rm -rf slurm_logs/
export EDPOSE_COCO_PATH=$WORK_DIR/sensoryArt_coco
## create environment
echo "creating environment"

# python3 -m venv venv
source venv/bin/activate
# cd models/edpose/ops
# rm -rf build/
# rm -rf dist/
# rm -rf MultiScaleDeformableAttention.egg-info/
# python setup.py build install
# cd ../../..
# python test.py
cd $WORK_DIR/ED-Pose-Gestures/

epoch=100
LR=0.0001
WEIGHT_DECAY=0.1
LR_DROP=30
NUM_GROUP=100
DN_NUMBER=100
N_QUERIES=900
N_CLASSES=17
# Create a run name with the combination of defined LR, weight_decay, num_group, etc.
commands=()
N=1

for ((i=0; i<N; i++))
do
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_extratoken"
    # Run the command with the random values and add it to the commands array
    command="python main.py  --seperate_token_for_class --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
        --output_dir logs/results_09_07/extratoken_finetune$i/all_coco/ \
        --options modelname=edpose num_classes=$N_CLASSES batch_size=4 epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco --find_unused_params \
        --finetune_edpose \
        --fix_size \
        --note $run_name"
    
    commands+=("$command")

done

N=1
WEIGHT_DECAY=0.01

for ((i=0; i<N; i++))
do
    run_name="lr${LR}_wd${WEIGHT_DECAY}_ng${NUM_GROUP}_dn${DN_NUMBER}_extratoken"
    # Run the command with the random values and add it to the commands array
    command="python main.py  --seperate_token_for_class --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
        --output_dir logs/results_09_07/extratoken_finetune$i/all_coco/ \
        --options modelname=edpose num_classes=$N_CLASSES batch_size=4 epochs=$epoch lr_drop=$LR_DROP lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER num_queries=$N_QUERIES num_group=$NUM_GROUP \
        --dataset_file=coco --find_unused_params \
        --finetune_edpose \
        --fix_size \
        --note $run_name"
    
    commands+=("$command")

done

eval ${commands[1]}
# srun ${commands[$SLURM_ARRAY_TASK_ID]}
