#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=sniffyArt
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
export EDPOSE_COCO_PATH=$WORK_DIR/coco_directory_gestures
## create environment
echo "creating environment"
source venv/bin/activate

commands=(
    # original
"python main.py --num_classes 7  --config_file config/edpose.cfg.py --pretrain_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --finetune_ignore class_embed. \
    --output_dir logs/train_sniffy_selected/edpose_original/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco"

    # classifier2
"python  main.py --num_classes 7  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy_selected/classifier2/deformable_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --edpose_finetune_ignore class_embed. \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params" 

"python  main.py --num_classes 7  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy_selected/classifier2/vanilla_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --edpose_finetune_ignore class_embed. \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

    # classifier 1

"python  main.py --num_classes 7  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy_selected/classifier1/deformable_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --edpose_finetune_ignore class_embed. \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params" 

"python  main.py --num_classes 7  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy_selected/classifier1/vanilla_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --edpose_finetune_ignore class_embed. \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

    # extra token
"python main.py --num_classes 7  --seperate_token_for_class --config_file config/edpose.cfg.py --pretrain_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --finetune_ignore class_embed. \
    --output_dir logs/train_sniffy_selected/extratoken/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco"
)

srun ${commands[$SLURM_ARRAY_TASK_ID]}
