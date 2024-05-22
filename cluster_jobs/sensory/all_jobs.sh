#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=sensoryArt
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,v100
#SBATCH --array=0-18  # Adjust based on the number of experiments
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
"python main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/classifier1/vanilla_finetune/all_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/classifier1/deformable_finetune/all_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier1/deformable_nofinetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --edpose_finetune_ignore class_embed. \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier1/deformable_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --edpose_finetune_ignore class_embed. \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params" 

"python main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier1/vanilla_nofinetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --edpose_finetune_ignore class_embed. \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier1/vanilla_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --edpose_finetune_ignore class_embed. \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/classifier2/vanilla_finetune/all_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/classifier2/deformable_finetune/all_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier2/deformable_nofinetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --edpose_finetune_ignore class_embed. \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier2/deformable_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --edpose_finetune_ignore class_embed. \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params" 

"python main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier2/vanilla_nofinetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco \
    --edpose_finetune_ignore class_embed. \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --seperate_classifier --classifier_type full --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/classifier2/vanilla_finetune/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --edpose_finetune_ignore class_embed. \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python main.py --num_classes 17  --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
--output_dir logs/train_sensory/edpose_original/all_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
--dataset_file=coco"
    
"python main.py --num_classes 17  --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
--output_dir logs/train_sensory/edpose_original/person_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
--dataset_file=coco \
--person_only"

"python main.py --num_classes 17  --config_file config/edpose.cfg.py --pretrain_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --finetune_ignore class_embed. \
--output_dir logs/train_sensory/edpose_original/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
--dataset_file=coco"

"python main.py --num_classes 17  --dec_layers 8 --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
--output_dir logs/train_sensory/edpose_nd8/all_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
--dataset_file=coco"

"python main.py --num_classes 17  --dec_layers 8 --config_file config/edpose.cfg.py --pretrain_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --finetune_ignore class_embed. \
--output_dir logs/train_sensory/edpose_nd8/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
--dataset_file=coco"

"python main.py --num_classes 17 --seperate_token_for_class --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
--output_dir logs/train_sensory/extratoken/all_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
--dataset_file=coco"

"python main.py --num_classes 17  --seperate_token_for_class --config_file config/edpose.cfg.py --pretrain_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --finetune_ignore class_embed. \
--output_dir logs/train_sensory/extratoken/all_edpose/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
--dataset_file=coco"
)

# echo ${commands[0]}
srun ${commands[$SLURM_ARRAY_TASK_ID]}

