#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --job-name=ed_actionpose
#SBATCH --gres=gpu:a100:4
#SBATCH --array=0-4 # Adjust based on the number of experiments
#SBATCH --output=/home/atuin/b268dc/b268dc10/logs/ed-actionpose/reproduction/%x_%j_%a.txt
#SBATCH --error=/home/atuin/b268dc/b268dc10/logs/ed-actionpose/reproduction/%x_%j_%a.txt

set -e

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=https://proxy:80


readonly GROUP=$(id -gn)

# COPY CODE DIRECTORY TO COMPUTE NODE
readonly CODE_SOURCE=/home/hpc/$GROUP/$USER/work/code/ED-Pose-Gestures
readonly TARGET_PATH=${TMPDIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

mkdir -p "${TARGET_PATH}"


echo "[$(date)] Copying code from ${CODE_SOURCE} to ${TARGET_PATH}"
cp -r ${CODE_SOURCE} ${TARGET_PATH}
cd ${TARGET_PATH}/ED-Pose-Gestures
echo "[$(date)] Code successfully copied."


# COPY DATA TO COMPUTE NODE
readonly SOURCE_DATA=/home/atuin/$GROUP/$USER/data/sensoryart/crossval/fold${SLURM_ARRAY_TASK_ID}.tar
readonly TARGET_DATA=${TARGET_PATH}/data 


mkdir -p "${TARGET_DATA}"
echo "[$(date)] Copying data from ${SOURCE_DATA} to ${TARGET_DATA}"
tar xf ${SOURCE_DATA} -C ${TARGET_DATA} --strip-components=1 # remove outer foldX directory to comply with mmdetection config expectations
echo "[$(date)] Data successfully copied."

export EDPOSE_COCO_PATH=${TARGET_DATA}

source "/home/atuin/${GROUP}/${USER}/venvs/edpose/bin/activate"

python - <<'PY'
import json, sys, time
paths = [
    "../data/annotations/person_keypoints_val2017.json",
    "../data/annotations/person_keypoints_test2017.json",
]
for p in paths:
    with open(p, "r") as f:
        d = json.load(f)
    if "info" not in d:
        d["info"] = {
            "description": "SensoryArt crossval",
            "version": "1.0",
            "year": 2025,
            "date_created": time.strftime("%Y-%m-%d")
        }
    if "licenses" not in d:
        d["licenses"] = []
    with open(p, "w") as f:
        json.dump(d, f)
    print(f"Patched {p}")
PY

# compile MSDA
module load gcc/11.2
module load cuda/11.7
cd models/edpose/ops
source make.sh
cd ../../../


# TRAINING
readonly WORK_DIR="${TARGET_PATH}/work_dir"
mkdir -p "${WORK_DIR}"

# Save console output to workdir
LOG_FILE="${WORK_DIR}/console_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging console output to: $LOG_FILE"


epoch=200
LR=0.0001
WEIGHT_DECAY=0.01
LR_DROP=30
NUM_GROUP=100
DN_NUMBER=100
N_QUERIES=900
BS=4
N_CLASSES=17

# Create a run name with the combination of defined LR, weight_decay, num_group, etc.
# export EDPOSE_COCO_PATH=${TARGET_DATA}/stratified_folds_unique_margin_8_n5/fold_${SLURM_ARRAY_TASK_ID}

CURRENT_PORT=$((44144+${SLURM_ARRAY_TASK_ID}))

torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$CURRENT_PORT main.py \
        --seperate_classifier --classifier_type full --config_file config/edpose.cfg.py \
        --seperate_token_for_class \
        --edpose_model_path /home/atuin/b268dc/b268dc10/models/EDPose-R50.pth \
        --edpose_finetune_ignore class_embed. \
        --output_dir ${WORK_DIR}/output/ \
        --options modelname=classifier \
            num_classes=$N_CLASSES batch_size=$BS epochs=$epoch lr_drop=$LR_DROP \
            lr=$LR weight_decay=$WEIGHT_DECAY lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
            set_cost_class=2.0 cls_loss_coef=2.0 use_dn=True dn_number=$DN_NUMBER \
            num_queries=$N_QUERIES num_group=$NUM_GROUP \
            name=edpose \
        --dataset_file=coco --find_unused_params \
        --finetune_edpose \
        --fix_size \
        --find_unused_params 

TARGET_WORKDIR="$WORK/work_dirs/ed_actionpose/full/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Training finished, start copying results to ${TARGET_WORKDIR}"

# COPY ANNOTATIONS FILES TO MAP CROSSVAL SPLIT
cp ../data/annotations/person_keypoints_train2017.json ${WORK_DIR}/output/
cp ../data/annotations/person_keypoints_val2017.json ${WORK_DIR}/output/
cp ../data/annotations/person_keypoints_test2017.json ${WORK_DIR}/output/

# COPY OUTPUT TO $WORK
mkdir -p "${TARGET_WORKDIR}"
cat "$0" > ${TARGET_WORKDIR}/slurm.sh # copy this file to workdir
cp -r ${WORK_DIR} ${TARGET_WORKDIR}
