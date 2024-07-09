WORK_DIR=/home/woody/iwi5/iwi5197h
export PYTHONPATH=${venv}/bin/python 
cd $WORK_DIR/ED-Pose-Simplified/
#rm -rf slurm_logs/
export Inference_Path=$WORK_DIR/sensoryArt_coco
export EDPOSE_COCO_PATH=$WORK_DIR/sensoryArt_coco
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

## create environment

# python3 -m venv venv
source venv/bin/activate


python  main.py \
 --output_dir "logs/edpose_original_new/"  \
 -c config/edpose.cfg.py \
 --options batch_size=4 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
 --dataset_file="coco" \
 --pretrain_model_path "logs/edpose_original_new/edpose_finetune0/all_coco/checkpoint.pth" \
 --eval