export EDPOSE_COCO_PATH=/net/cluster/azhar/datasets/gestuers_keypoints_cmac/smell-gesture-recognition/coco_directory_gestures
#   python -m torch.distributed.launch --nproc_per_node=4  main.py \
python main.py  --seperate_classifier --config_file "config/classifier.cfg.py" --edpose_model_path "logs/train/gestures_allclasses_coco_pretrained_r50/checkpoint.pth" \
    --output_dir "logs/train/gestures_classifier1/finetune2_deformable/all_classes_pretrained_r50_coco/" --options batch_size=2 epochs=20 lr_drop=6 num_body_points=17 backbone='resnet50' \
    --classifier_use_deformable \
    --dataset_file="coco" \
    --classifier_decoder_layers=2 \
