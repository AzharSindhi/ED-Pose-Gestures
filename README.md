# Installation
Please visit the original repo for the initial setup and installation: [ED-Pose](https://github.com/IDEA-Research/ED-Pose). 

# Key Changes
- We add the support of adding separate gesture-specific queries used for the classification through additional input argument `--seperate_token_for_class`.
- Similarly, we provide the support of a combined classifier variant through the additional argument `--seperate_classifier`. By default the combined classifier variant uses the vanilla cross attention, if you want to use the deformable one, pass the argument `classifier_use_deformable` additionally.

# Training
For example, to train the combined classifier version on the sensoryArt dataset, you can run the following script:

```
export EDPOSE_COCO_PATH=/path/to/dataset
python main.py  --seperate_classifier --config_file config/edpose.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
        --output_dir logs/multiruns_stratified_21_07/vanilla_full_noextra$i/all_coco/ \
        --options modelname=classifier num_classes=17 batch_size=4 epochs=100 lr_drop=30 lr=0.0001 weight_decay=0.01 lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 \
        --dataset_file=coco --find_unused_params \
        --finetune_edpose \
        --fix_size \
        --note combined_classifier_run   
```

To train the model with extra class queries, please run:


```
export EDPOSE_COCO_PATH=/path/to/dataset

python main.py  --seperate_token_for_class --config_file config/edpose.cfg.py --edpose_model_path ./models/edpose_r50_coco.pth --edpose_finetune_ignore class_embed. \
        --output_dir logs/multiruns_stratified_21_07/extratoken_finetuned$i/all_coco/ \
        --options modelname=edpose num_classes=17 batch_size=4 epochs=100 lr_drop=30 lr=0.0001 weight_decay=0.01 lr_backbone=1e-05 num_body_points=17 backbone=resnet50 \
        set_cost_class=2.0 cls_loss_coef=2.0 \
        --dataset_file=coco --find_unused_params \
        --finetune_edpose \
        --fix_size \
        --note extratokens_run
```

# Reproduciblity
Please run the provided script for training on the stratified folds: `scripts/multiruns_stratified.sh`.

# Output format
The runs are directly saved under `output_dir` provided above. Moreover, the results are written to the mlflow as well for visualization. 