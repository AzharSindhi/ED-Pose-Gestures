#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=sensoryArt
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,v100
#SBATCH --array=0-5  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_err.txt

# export SLURM_NPROCS=1
export PYTHONPATH=${venv}/bin/python 
set -e # exit on error to prevent crazy stuff form happening unnoticed

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

commands=(
    # original
"python main.py --num_classes 17  --config_file config/edpose.cfg.py --pretrain_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --finetune_ignore class_embed. \
    --output_dir logs/train_sensory_selected_new/edpose_original/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco"

    # classifier2
"python  main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory_selected_new/classifier2/deformable_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --edpose_finetune_ignore class_embed. \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params" 

"python  main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory_selected_new/classifier2/vanilla_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --edpose_finetune_ignore class_embed. \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

    # classifier 1

"python  main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory_selected_new/classifier1/deformable_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --edpose_finetune_ignore class_embed. \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params" 

"python  main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory_selected_new/classifier1/vanilla_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --edpose_finetune_ignore class_embed. \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

    # extra token
"python main.py --num_classes 17  --seperate_token_for_class --config_file config/edpose.cfg.py --pretrain_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --finetune_ignore class_embed. \
    --output_dir logs/train_sensory_selected_new/extratoken/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco"
)

# srun ${commands[$SLURM_ARRAY_TASK_ID]}
for command in "${commands[@]}"
do
    # echo $command
    eval $command
done
