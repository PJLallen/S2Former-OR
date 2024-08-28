import argparse
import datetime
import random
import time
from pathlib import Path
from src.engine.evaluator_or import or_evaluate_infer
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


import src.util.misc as utils
from src.engine.arg_parser import get_args_parser
from src.data.datasets import build_dataset
from src.engine.trainer import train_one_epoch
from src.engine import or_evaluate
from src.models import build_model

from src.util.logger import print_params, print_args
from collections import OrderedDict

def save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename):
    # save_ckpt: function for saving checkpoints
    output_dir = Path(args.output_dir)
    if args.output_dir:
        checkpoint_path = output_dir / f'{filename}.pth'
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

def main(args):
    utils.init_distributed_mode(args)

    if not args.train_detr is not None: # pretrained DETR
        print("Freeze weights for detector")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.num_classes = 11
    args.num_actions = 14
    args.action_names = ["Assisting", "Cementing", "Cleaning", "CloseTo", "Cutting", "Drilling", "Hammering", "Holding", "LyingOn", "Operating", "Preparing", "Sawing", "Suturing", "Touching", "None"]

    # Data Setup
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_infer = build_dataset(image_set='infer', args=args)


    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers
    print_args(args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_infer = DataLoader(dataset_infer, args.batch_size, collate_fn=utils.collate_fn)


    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = print_params(model)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "detr" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("detr" in n and 'backbone' not in n) and p.requires_grad],
            "lr": args.lr * 0.1,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("detr" in n and 'backbone' in n) and p.requires_grad],
            "lr": args.lr * 0.01,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.reduce_lr_on_plateau_factor, patience=args.reduce_lr_on_plateau_patience, verbose=True)

    # Weight Setup
    if args.detr_weights is not None:
        print(f"Loading detr weights from args.detr_weights={args.detr_weights}")
        if args.detr_weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.detr_weights, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.detr_weights, map_location='cpu')

        if 'hico_ft_q16.pth' in args.detr_weights: # hack: for loading hico fine-tuned detr
            mapped_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k.startswith('detr.'):
                    mapped_state_dict[k.replace('detr.', '')] = v
            model_without_ddp.detr.load_state_dict(mapped_state_dict)
        else:
            model_without_ddp.detr.load_state_dict(checkpoint['model'], strict=False)

    if args.resume:
        print(f"Loading model weights from args.resume={args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.infer:
        # infer only mode for 4D-OR
        or_evaluate_infer(model, postprocessors, data_loader_infer, device, 0, args)
        return

    if args.eval:
        total_res = or_evaluate(model, postprocessors, data_loader_val, device, args=args, thr=0)
        return

    # Training starts here!
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.epochs,
            args.clip_max_norm, dataset_file=args.dataset_file)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR): lr_scheduler.step()

        # Validation
        if args.validate and epoch >= 40:
            print('-'*100)
            total_res = or_evaluate(model, postprocessors, data_loader_val, device, args=args, thr=0)
        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename=f'checkpoint_{epoch}')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    # training
    parser.add_argument('--detr_weights', default=None, type=str)
    parser.add_argument('--train_detr', action='store_true', default=False)
    parser.add_argument('--finetune_detr_weight', default=0.1, type=float)
    parser.add_argument('--lr_detr', default=1e-5, type=float)
    parser.add_argument('--reduce_lr_on_plateau_patience', default=2, type=int)
    parser.add_argument('--reduce_lr_on_plateau_factor', default=0.1, type=float)
    parser.add_argument('--valid_obj_ids', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)

    # loss
    parser.add_argument('--proposal_focal_loss_alpha', default=0.75, type=float) # large alpha for high recall
    parser.add_argument('--action_focal_loss_alpha', default=0.5, type=float)
    parser.add_argument('--proposal_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--action_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--proposal_loss_coef', default=1, type=float)
    parser.add_argument('--action_loss_coef', default=1, type=float)

    # ablations
    parser.add_argument('--no_hard_mining_for_relation_discovery', dest='use_hard_mining_for_relation_discovery', action='store_false', default=True)
    parser.add_argument('--no_relation_dependency_encoding', dest='use_relation_dependency_encoding', action='store_false', default=True)
    parser.add_argument('--no_memory_layout_encoding', dest='use_memory_layout_encoding', action='store_false', default=True, help='layout encodings')
    parser.add_argument('--no_nms_on_detr', dest='apply_nms_on_detr', action='store_false', default=True)
    parser.add_argument('--no_tail_semantic_feature', dest='use_tail_semantic_feature', action='store_false', default=True)
    parser.add_argument('--no_spatial_feature', dest='use_spatial_feature', action='store_false', default=True)
    parser.add_argument('--no_interaction_decoder', action='store_true', default=False)
    # multiview arguments
    parser.add_argument('--use_multiviewfusion', action='store_true', default=False)
    parser.add_argument('--use_view6', action='store_true', default=False)
    parser.add_argument('--use_head_semantic_feature', action='store_true', default=False)
    parser.add_argument('--use_multiviewfusion_last', action='store_true', default=False)
    parser.add_argument('--use_multiviewfusion_last_view2', action='store_true', default=False)
    parser.add_argument('--use_multiviewfusion_last_all', action='store_true', default=False)
    parser.add_argument('--deformable_detr', action='store_true', default=False)
    parser.add_argument('--visualization', action='store_true', default=False)

    # point cloud arguments
    parser.add_argument('--use_pointsfusion', action='store_true', default=False)
    parser.add_argument('--use_simple_pointsfusion', action='store_true', default=False)
    parser.add_argument('--additional_encoder', action='store_true', default=False)

    # prior embedding
    parser.add_argument('--use_prior', action='store_true', default=False)
    parser.add_argument('--use_tricks', action='store_false', default=True)
    parser.add_argument('--use_tricks_val', action='store_false', default=True)
    parser.add_argument('--add_none', action='store_false', default=True)

    # not sensitive or effective
    parser.add_argument('--use_memory_union_mask', action='store_true', default=False)
    parser.add_argument('--use_union_feature', action='store_true', default=False)
    parser.add_argument('--adaptive_relation_query_num', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask_attend_topk', default=10, type=int)
    parser.add_argument('--use_prior_verb_label_mask', action='store_true', default=False)
    parser.add_argument('--relation_feature_map_from', default='backbone', help='backbone | detr_encoder')
    parser.add_argument('--use_query_fourier_encoding', action='store_true', default=False)
    parser.add_argument('--img_folder', default='data/4dor/images/',
                        help='path')
    parser.add_argument('--img_folder_infer', default='data/4dor/infer/',
                        help='path')
    parser.add_argument('--ann_path', default='data/4dor/',
                        help='path')
    # for infer
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--infer_val', action='store_true', default=False)
    parser.add_argument('--infer_name', default='infer_or.json',
                        help='infer json save path')
    parser.add_argument('--closeto', default=0.12, type=float)
    parser.add_argument('--num_feature_levels', default=1, type=int)


    args = parser.parse_args()
    args.STIP_relation_head = True

    if args.output_dir:
        args.output_dir += f"/{args.group_name}/{args.run_name}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
