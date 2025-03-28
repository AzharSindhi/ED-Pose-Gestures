import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
from typing import Optional
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch,inference_vis
import models
from util.config import DictAction, Config
from util.utils import ModelEma, BestMetricHolder
import mlflow
from pathlib import Path
import time
# from dotenv import load_dotenv
# load_dotenv()

# torch.backends.cudnn.benchmark = False # for reproducibility
# torch.use_deterministic_algorithms(True, warn_only=True) # for reproducibility

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, default="config/edpose.cfg.py")#required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')


    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default=None)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--sanity', action='store_true')


    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--decoder_box_layers', default=2, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # temporary #
    parser.add_argument('--box_detach_type', default="None")
    parser.add_argument('--class_detach_type', default="None")
    parser.add_argument('--decoder_class_detach', default="None")
    
    # distributed training parameters
    parser.add_argument('--no_distributed', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--person_only', action='store_true',
                        help="Train only person class")
    parser.add_argument('--no_dn', action='store_true',
                        help="Training using no denoising")
    
    # when there is a seperate classifier or token (modified)
    parser.add_argument("--seperate_token_for_class", action='store_true', help="fine tuning using seperate class token")
    parser.add_argument("--seperate_classifier", action='store_true', help="fine tuning using seperate decoder for classes")
    parser.add_argument('--classifier_type', type=str, choices=["full", "partial"], default="full", help='specifiy the classifier type')
    parser.add_argument('--edpose_model_path', type=str, help='load edpose from other checkpoint')
    parser.add_argument('--edpose_finetune_ignore', type=str, nargs='+', help="which keys to ignore in the weights dictionary?")
    parser.add_argument("--finetune_edpose", action='store_true', help="whether to finetune edpose or used saved weights")
    parser.add_argument("--classifier_decoder_return_intermediate", action='store_true', help="return intermediate outputs from classifier decoder")
    parser.add_argument("--classifier_use_deformable", action='store_true', help="use deformable DETR for classifier, otherwise the Vanilla decoder will be used")
    parser.add_argument("--classifier_decoder_layers", type=int, default=2, help="number of decoder layers for classifier")
    parser.add_argument("--use_cls_token", action='store_true', help="use class token for classifier")
    parser.add_argument("--use_clip_prior", action='store_true', help="use class token for classifier")
    parser.add_argument("--use_class_prior", action='store_true', help="use class token for classifier")
    parser.add_argument("--queries_transform", action='store_true', help="use class token for classifier")

    return parser


# mlflow.set_tracking_uri(uri="http://127.0.0.1:5002") 

def set_mlflow_run(args, output_dir):

    path_parts = output_dir.split("/")
    if path_parts[-1] == "":
        path_parts = path_parts[:-1]

    experiment = path_parts[-3]
    # mlflow.set_experiment(experiment)

    is_finetuned = ["no_finetuned", "finetuned"]
    run_name = f"{args.note}_{path_parts[-2]}_{path_parts[-1]}"
    return experiment, run_name

def log_metric_to_mlflow(metrics):
    """
    include_strings ignore if include all is true
    """
    include = ["lr", "class_error", "loss", "loss_ce", 
               "loss_bbox", "loss_giou", "loss_keypoints", 
               "loss_oks", "coco_eval_bbox", 
               "coco_eval_keypoints_detr"]
    
    timenow = time.time()
    for key, value in metrics.items():
        if "_".join(key.split("_")[1:]) in include:
            if isinstance(value, list):
                [mlflow.log_metric(f"{key}_{i}", round(float(v), 2)) for i, v in enumerate(value)]
            
            else:
                if "lr" not in key:
                    value = round(float(value), 2)
                mlflow.log_metric(key, value)
    
    print("-------logging takes:-----------", time.time() - timenow)

def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    if args.no_distributed:
        args.distributed = False
    else:
        utils.init_distributed_mode(args)
    
    time.sleep(np.random.randint(1, 5)) # to avoid multiple processes writing to the same file
    # args.distributed=False
    print("Loading config file from {}".format(args.config_file))
    output_dir_path = args.output_dir
    # rest of the code...
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False
    if args.dataset_file=="coco":
        args.coco_path = os.environ.get("EDPOSE_COCO_PATH")
    elif args.dataset_file=="crowdpose":
        args.crowdpose_path = os.environ.get("EDPOSE_CrowdPose_PATH")
    elif args.dataset_file=="humanart":
        args.humanart_path = os.environ.get("EDPOSE_HumanArt_PATH")

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        # print("args:", vars(args))
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed #+ utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    # dataset_test = build_dataset(image_set='test', args=args)
    imset = "val"
    args.class_names = dataset_train.class_names

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)


    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        # sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        # sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)



    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    # args.output_dir = ""
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        


    if args.eval:
        if os.environ.get("Inference_Path"):
            inference_vis(model, criterion, postprocessors,
                     data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
            return

        else:
            os.environ['EVAL_FLAG'] = 'TRUE'
            val_stats, val_coco_evaluator, predictions_json_box, predictions_json_kps = evaluate(model, criterion, postprocessors,
                                                  data_loader_test, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args, img_set="test")
            if args.output_dir:
                utils.save_on_master(val_coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

            log_stats = {**{f'test_{k}': v for k, v in val_stats.items()} }
            
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                with open(os.path.join(output_dir, 'bbox_predictions_test.json'), 'w') as f:
                    json.dump(predictions_json_box, f)
                with open(os.path.join(output_dir, 'keypoints_predictions_test.json'), 'w') as f:
                    json.dump(predictions_json_kps, f)
            return

    args.optimizer = str(optimizer)
    experiment, run_name = set_mlflow_run(args, output_dir_path)
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        # mlflow.set_tag("mlflow.runName", f"{run_name}")
        mlflow.log_params(vars(args))
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']

            if not args.onecyclelr:
                lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                #     checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': ema_m.module.state_dict(),
                        })
                    utils.save_on_master(weights, checkpoint_path)
                    
            # val
            val_stats, val_coco_evaluator, predictions_json_box, predictions_json_kps = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None), img_set=imset
            )
            #test
            # test_stats, test_coco_evaluator = evaluate(
            #     model, criterion, postprocessors, data_loader_test, base_ds, device, args.output_dir,
            #     wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None),
            #                                                       img_set="test"
            # )
            map_regular = val_stats["coco_eval_keypoints_detr"][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                with open(os.path.join(output_dir, 'bbox_predictions.json'), 'w') as f:
                    json.dump(predictions_json_box, f)
                with open(os.path.join(output_dir, 'keypoints_predictions.json'), 'w') as f:
                    json.dump(predictions_json_kps, f)
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                # **{f'test_{k}': v for k, v in test_stats.items()},
            }
            log_metric_to_mlflow(log_stats)

            # eval ema
            if args.use_ema:
                ema_val_stats, ema_val_coco_evaluator = evaluate(
                    ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None), img_set=imset
                )
                log_stats.update({f'ema_test_{k}': v for k,v in ema_val_stats.items()})
                map_ema = ema_val_stats['coco_eval_keypoints_detr'][0]
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    utils.save_on_master({
                        'model': ema_m.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            log_stats.update(best_map_holder.summary())

            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass

            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if val_coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in val_coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(val_coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
