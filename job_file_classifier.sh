#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=classifier_experiments
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
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
export EDPOSE_COCO_PATH=$WORK_DIR/coco_directory_gestures
## create environment
echo "creating environment"

# python3 -m venv venv
source venv/bin/activate
# pip3 install torch torchvision
# pip install -r requirements.txt

#cd models/edpose/ops
#python setup.py build install
# unit test (should see all checking is True)
#python test.py
#cd ../../..

echo "created environment"
# Read and parse commands from the bash script
commands=()  # Initialize an empty array to hold the commands
current_command=""  # Temporary string to hold commands
should_append=false  # Control flag for appending lines

 while IFS='' read -r line || [[ -n "$line" ]]; do
   # Check if the line starts a new command
   if [[ "$line" =~ ^python ]]; then
     if [[ -n "$current_command" ]]; then
       cleaned_command=$(echo "$current_command" | sed 's/\\//g; s/  */ /g')
       commands+=("$cleaned_command")  # Save the previous command before starting a new one
       # echo "$current_command"  # Print the command
     fi
     current_command="$line"  # Start new command
     should_append=true  # Set appending flag
   elif [[ "$should_append" = true ]]; then
     # Append the line to the current command if it's not empty or a comment
     if [[ -n "$line" && ! "$line" =~ ^# ]]; then
       current_command+=" $line"
     fi
   fi
 done < run_classifier_experiments.sh # bash script file

 # Add the last command if it exists
 if [[ -n "$current_command" ]]; then
    cleaned_command=$(echo "$current_command" | sed 's/\\//g; s/  */ /g')
    commands+=("$cleaned_command")
 fi

 #Number of jobs is the number of commands
 num_jobs=${#commands[@]}
 echo "$num_jobs jobs will be submitted."

srun ${commands[$SLURM_ARRAY_TASK_ID]}

