# export EDPOSE_COCO_PATH=/net/cluster/azhar/datasets/gestuers_keypoints_cmac/smell-gesture-recognition/coco_directory_gestures
#   python -m torch.distributed.launch --nproc_per_node=4  main.py \
#################################################### pretrained on coco #############################################################

# python main.py  --config_file config/edpose.cfg.py --pretrain_model_path ./models/edpose_r50_coco.pth --finetune_ignore class_embed. \
#     --output_dir logs/train_sniffy/without_classifier_without_token/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
#     --dataset_file=coco 
#batch_size=4
# WORK_DIR=/home/woody/iwi5/iwi5197h
# export PYTHONPATH=${venv}/bin/python 
# export EDPOSE_COCO_PATH=$WORK_DIR/coco_directory_gestures

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/vanilla_cocopretrained_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/deformable_cocopretrained_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \
#################################### finetuned on already trained EDpose for all classes ###########################

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_allclasses_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/deformable_nofinetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_allclasses_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/deformable_finetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_allclasses_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/vanilla_nofinetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_allclasses_coco_pretrained_r50/checkpoint.pth \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/vanilla_finetune_nd2/all_classes_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \
###################################################### finetuned on person only ##################################################################################################

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/deformable_nofinetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/deformable_finetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --classifier_use_deformable \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/vanilla_nofinetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params \

python  main.py  --seperate_classifier --config_file config/classifier.cfg.py --edpose_model_path logs/train_sniffy/gestures_persononly_coco_pretrained_r50/checkpoint.pth --edpose_finetune_ignore class_embed. \
    --output_dir logs/train_sniffy/gestures_classifier_multiple/vanilla_finetune_nd2/person_only_pretrained_r50_coco/ --options batch_size=4 epochs=20 lr_drop=6 num_body_points=17 backbone=resnet50 \
    --finetune_edpose \
    --dataset_file=coco \
    --classifier_decoder_layers=2 --find_unused_params
