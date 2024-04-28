#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=classifier_experiments
#SBATCH --gres=gpu:1
#SBATCH --output=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/ED-Pose-Gestures/slurm_logs/%x_%j_err.txt

set -e # exit on error to prevent crazy stuff form happening unnoticed

# module load python 
# module load cuda
WORK_DIR=/home/woody/iwi5/iwi5197h
cd $WORK_DIR/ED-Pose-Gestures/
rm -rf slurm_logs/
export EDPOSE_COCO_PATH=$WORK_DIR/coco_directory_gestures
## create environment
echo "creating environment"

# python3 -m venv venv
source venv/bin/activate
# pip3 install torch torchvision
# pip install -r requirements.txt
cd models/edpose/ops
python setup.py build install
# unit test (should see all checking is True)
#python test.py
cd ../../..

echo "created environment"
# Read and parse commands from the bash script
commands=()  # Initialize an empty array to hold the commands
current_command=""  # Temporary string to hold commands
should_append=false  # Control flag for appending lines

# while IFS='' read -r line || [[ -n "$line" ]]; do
#   # Check if the line starts a new command
#   if [[ "$line" =~ ^python\ main\.py ]]; then
#     if [[ -n "$current_command" ]]; then
#       commands+=("$current_command")  # Save the previous command before starting a new one
#       # echo "$current_command"  # Print the command
#     fi
#     current_command="$line"  # Start new command
#     should_append=true  # Set appending flag
#   elif [[ "$should_append" = true ]]; then
#     # Append the line to the current command if it's not empty or a comment
#     if [[ -n "$line" && ! "$line" =~ ^# ]]; then
#       current_command+=" $line"
#     fi
#   fi
# done < $1 # bash script file

# # Add the last command if it exists
# if [[ -n "$current_command" ]]; then
#   commands+=("$current_command")
# fi

# Number of jobs is the number of commands
# num_jobs=${#commands[@]}
# echo "$num_jobs jobs will be submitted."
# # Launch multiple experiments in parallel
# num_jobs=1
# for ((i=0; i<$num_jobs; i++))
# do
#     echo "COMMAND: ${COMMANDS[$i]}"
#     sbatch ${COMMANDS[$i]}
# done
python main.py  --seperate_classifier --config_file "config/classifier.cfg.py" --edpose_model_path "logs/train/gestures_allclasses_coco_pretrained_r50/checkpoint.pth" \
    --output_dir "logs/train/gestures_classifier_multiple/deformable_finetune_nd2/all_classes_pretrained_r50_coco/" --options batch_size=8 epochs=20 lr_drop=6 num_body_points=17 backbone='resnet50' \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file="coco" \
    --classifier_decoder_layers=2

deactivate

