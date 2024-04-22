# export Inference_Path=/net/cluster/azhar/datasets/gestuers_keypoints_cmac/smell-gesture-recognition/images
# export EDPOSE_COCO_PATH=/net/cluster/shared_dataset/COCO2017
#   python -m torch.distributed.launch --nproc_per_node=4  main.py \
python main.py
#  --output_dir "logs/coco_r50" \
#  -c config/edpose.cfg.py \
--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
--dataset_file="coco" \
--pretrain_model_path "./models/edpose_r50_humanart.pth" \
--eval