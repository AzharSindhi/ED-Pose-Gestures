export EDPOSE_COCO_PATH=/net/cluster/azhar/datasets/gestuers_keypoints_cmac/smell-gesture-recognition/coco_directory_gestures
#   python -m torch.distributed.launch --nproc_per_node=4  main.py \
python main.py --config_file "config/edpose.cfg.py" --output_dir "logs/train/gestures/finetune10/allclasses_coco_pretrained_r50/" --options batch_size=2 epochs=20 lr_drop=6 num_body_points=17 backbone='resnet50' --dataset_file="coco" --pretrain_model_path "./models/edpose_r50_coco.pth" --finetune_ignore "class_embed."