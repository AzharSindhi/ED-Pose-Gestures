#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=token_classifier_experiments
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,v100
#SBATCH --array=0-9  # Adjust based on the number of experiments
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
# export EDPOSE_COCO_PATH=$WORK_DIR/sensoryArt_coco
## create environment
echo "creating environment"

# python3 -m venv venv
source venv/bin/activate
# pip3 install torch torchvision

# export EDPOSE_COCO_PATH=/net/cluster/azhar/datasets/gestuers_keypoints_cmac/smell-gesture-recognition/coco_directory_gestures
#   python -m torch.distributed.launch --nproc_per_node=4  main.py \
#################################################### pretrained on coco #############################################################

# python main.py  --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
#     --output_dir logs/train_sensory/without_classifier_without_token/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
#     --dataset_file=coco 
#batch_size=4
# WORK_DIR=/home/woody/iwi5/iwi5197h
# export PYTHONPATH=${venv}/bin/python 
# export EDPOSE_COCO_PATH=$WORK_DIR/coco_directory_gestures

commands=(
"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/gestures_classifier_multiple/vanilla_cocopretrained_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/gestures_classifier_multiple/deformable_cocopretrained_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/deformable_nofinetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/deformable_finetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params" 

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/vanilla_nofinetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/vanilla_finetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/deformable_nofinetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/deformable_finetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/vanilla_nofinetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"

"python  main.py --num_classes 17  --seperate_classifier --seperate_token_for_class --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy_old/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sensory/gestures_extratoken_multiple/vanilla_finetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params"
)

# echo ${commands[0]}
srun ${commands[$SLURM_ARRAY_TASK_ID]}
