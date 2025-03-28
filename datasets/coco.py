"""
COCO dataset which returns image_id for evaluation.
"""
import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from numpy.core.defchararray import array
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import datasets.transforms_coco as T
from datasets.data_util import preparing_dataset
__all__ = ['build']

class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root_path, image_set, num_classes, transforms, return_masks, sanity=False, person_only=False):
        super(CocoDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, num_classes)
        self.Inference_Path = os.environ.get("Inference_Path")
        if person_only:
            assert num_classes == 1
            self.num_classes = 1
        if sanity:
            self.img_folder = root_path / "val2017"
            self.coco = COCO(root_path / "annotations/person_keypoints_val2017.json")
            self.all_imgIds = sorted(self.coco.getImgIds())[:10]
            # get category names
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            self.class_names = [cat['name'] for cat in cats]

            # for image_id in imgIds:
            #     if self.coco.getAnnIds(imgIds=image_id) == []:
            #         continue
            #     ann_ids = self.coco.getAnnIds(imgIds=image_id)
            #     target = self.coco.loadAnns(ann_ids)
            #     num_keypoints = [obj["num_keypoints"] for obj in target]
            #     if sum(num_keypoints) == 0:
            #         continue
                # self.all_imgIds.append(image_id)
            return
        
        if image_set == "train":
            self.img_folder = root_path / "train2017"
            self.coco = COCO(root_path / "annotations/person_keypoints_train2017.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:
                if self.coco.getAnnIds(imgIds=image_id) == []:
                    continue
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                target = self.coco.loadAnns(ann_ids)
                num_keypoints = [obj["num_keypoints"] for obj in target]
                if sum(num_keypoints) == 0:
                    continue
                self.all_imgIds.append(image_id)
        elif image_set == "val":
            self.img_folder = root_path / "val2017"
            self.coco = COCO(root_path / "annotations/person_keypoints_val2017.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:
                self.all_imgIds.append(image_id)
        else:
            if self.Inference_Path:
                self.all_file = []
                self.all_imgIds = []
                for file in os.listdir(self.Inference_Path):
                    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                        self.all_file.append(file)
                        self.all_imgIds.append(os.path.splitext(file)[0])
            else:
                self.img_folder = root_path / "test2017"
                self.coco = COCO(root_path / "annotations/person_keypoints_test2017.json")
                imgIds = sorted(self.coco.getImgIds())
                self.all_imgIds = []
                for image_id in imgIds:
                    self.all_imgIds.append(image_id)
        # get category names
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        self.class_names = [cat['name'] for cat in cats]

    def __len__(self):
        return len(self.all_imgIds)

    def __getitem__(self, idx):
        if self.Inference_Path:
            image_id = self.all_imgIds[idx]
            target = {'image_id': int(image_id), 'annotations': []}
            img = Image.open(self.Inference_Path + self.all_file[idx])
            img, target = self.prepare(img, target)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            return img, target
        else:
            image_id = self.all_imgIds[idx]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            target = self.coco.loadAnns(ann_ids)

            target = {'image_id': image_id, 'annotations': target}
            img = Image.open(self.img_folder / self.coco.loadImgs(image_id)[0]['file_name'])
            img, target = self.prepare(img, target)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, num_classes=17):
        self.return_masks = return_masks
        self.num_classes = num_classes

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        anno = [obj for obj in anno if obj['num_keypoints'] != 0]
        keypoints = [obj["keypoints"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        if self.num_classes == 1:
            classes = [0] * len(anno)
            
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        keypoints = keypoints[keep]
        if self.return_masks:
            masks = masks[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]
    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test']:


        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    dataset = CocoDetection(root, image_set, args.num_classes, transforms=make_coco_transforms(image_set, strong_aug=args.strong_aug),
                            return_masks=args.masks, sanity=args.sanity, person_only=args.person_only)

    return dataset

