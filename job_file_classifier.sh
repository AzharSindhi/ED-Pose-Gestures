#!/bin/bash -l
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1   
#SBATCH --mem-per-gpu=30G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export EDPOSE_COCO_PATH=$WORK/smell-gesture-recognition/coco_directory_gestures
cd $WORK/ED-Pose-Gestures/
# Read and parse commands from the bash script
commands=()  # Initialize an empty array to hold the commands
current_command=""  # Temporary string to hold commands
should_append=false  # Control flag for appending lines

while IFS='' read -r line || [[ -n "$line" ]]; do
  # Check if the line starts a new command
  if [[ "$line" =~ ^python\ main\.py ]]; then
    if [[ -n "$current_command" ]]; then
      commands+=("$current_command")  # Save the previous command before starting a new one
      echo "$current_command"  # Print the command
    fi
    current_command="$line"  # Start new command
    should_append=true  # Set appending flag
  elif [[ "$should_append" = true ]]; then
    # Append the line to the current command if it's not empty or a comment
    if [[ -n "$line" && ! "$line" =~ ^# ]]; then
      current_command+=" $line"
    fi
  fi
done < $1 # bash script file

# Add the last command if it exists
if [[ -n "$current_command" ]]; then
  commands+=("$current_command")
fi
# Number of jobs is the number of commands
num_jobs=${#commands[@]}
echo "$num_jobs jobs will be submitted."
# Launch multiple experiments in parallel
num_jobs=1
for ((i=0; i<$num_jobs; i++))
do
    sbatch --job-name="edpose_classifier1_$i" ${COMMANDS[$i]}
done

