# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import pytz
import time
import json
import argparse
import datetime
import subprocess
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import timm
assert timm.__version__ == "0.3.2"

from mmengine.config import DictAction

import models
import utils.misc as misc
from utils import DATASET_DICT
from engine import train_one_epoch, test_one_epoch
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


def set_output_dir(args):

    tz_beijing = pytz.timezone('Asia/Shanghai') 
    datetime_NY = datetime.datetime.now(tz_beijing)
    if args.output_dir:
        save_folder_name = f'exp/exp-{args.output_dir}-{datetime_NY.strftime("%Y%m%d%H%M%S")}'
    else:
        save_folder_name = f'exp/exp-debug-{datetime_NY.strftime("%Y%m%d%H%M%S")}'

    args.output_dir = save_folder_name
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MRM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mrm', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/dataset/mimic_cxr_ap-pa_dataset', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # add the mmengine config
    parser.add_argument('--aliyunpan_path', default='/exp_cross_modal_medical_retrieval', type=str)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    return parser

def main(args):

    misc.init_distributed_mode(args)

    if misc.is_main_process():

        set_output_dir(args)

        with open(f"{args.output_dir}/config.json", mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_resolution if hasattr(args, 'input_resolution') else 224 , scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
        transforms.RandomAffine(degrees=(-10.0, 10.0), translate=(0.01, 0.05), scale=(0.95, 1.05)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449])])
    dataset_train = DATASET_DICT[args.dataset]('./dataset/mimic-cxr/official_protocol_train.csv', 
                                               os.path.join(args.data_path), 
                                               transform=transform_train,
                                               mask_prob=args.report_mask_prob if hasattr(args, 'report_mask_prob') else 0.5,
                                               max_caption_length=128,
                                               token_name=args.report_token_name if hasattr(args, 'report_token_name') else "/workspace/Bio_ClinicalBERT")

    print(dataset_train)

    # simple augmentaion
    transform_validation = transforms.Compose([
        transforms.Resize([args.input_resolution if hasattr(args, 'input_resolution') else 224, 
                           args.input_resolution if hasattr(args, 'input_resolution') else 224], interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449])])
    dataset_validation = DATASET_DICT[args.dataset]('./dataset/mimic-cxr/official_protocol_validate.csv',
                                                    os.path.join(args.data_path),
                                                    transform=transform_validation,
                                                    mask_prob=args.report_mask_prob if hasattr(args, 'report_mask_prob') else 0.0,
                                                    max_caption_length=512,
                                                    token_name=args.report_token_name if hasattr(args, 'report_token_name') else "/workspace/Bio_ClinicalBERT")
    
    print(dataset_validation)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_validation = torch.utils.data.DistributedSampler(
            dataset_validation, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_validation = %s" % str(sampler_validation))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=f'{args.output_dir}/runs')
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_train.collate_fn
    )

    data_loader_validation = torch.utils.data.DataLoader(
        dataset_validation, sampler=sampler_validation,
        batch_size=96,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=dataset_validation.collate_fn, 
        shuffle=False
    )

    # TODO: Determine a unique image-report pairs ([view1, view2, ..., viewN ; report])
    identity_dict = {}
    for index, path in enumerate(data_loader_validation.dataset.images_list):
        
        if "_".join(path.split('/')[1:3]) not in identity_dict.keys():
            identity_dict["_".join(path.split('/')[1:3])] = [index]
        else:
            identity_dict["_".join(path.split('/')[1:3])].append(index)

    image_identity_labels = []
    for path in data_loader_validation.dataset.images_list:
        image_identity_labels.append(identity_dict["_".join(path.split('/')[1:3])])

    report_identity_indexes = set([identity_dict[key][0] for key in identity_dict.keys()])

    # define the model
    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)
    
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model._set_static_graph()
        model_without_ddp = model.module
    
    optimizer = torch.optim.AdamW(model.module.get_parameter_group(), lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        if misc.is_main_process():
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # evaluate set 
        if epoch % 1 == 0 and args.output_dir:
            infos = test_one_epoch(model, data_loader_validation, f"cuda:{global_rank}", 
                                   epoch, image_identity_labels, report_identity_indexes,
                                   log_writer=log_writer, args=args)
            
            if log_writer is not None:
                log_writer.flush()

            if misc.is_main_process():
                print_info = ""
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(f'[epoch {epoch}]')
                    print_info += f'[epoch {epoch}]'
                    for key in infos.keys():
                        f.write(f' | {key} - i2t: {infos[key]["i2t"]["topK(1)"]}, {infos[key]["i2t"]["topK(5)"]}, {infos[key]["i2t"]["topK(10)"]}; t2i: {infos[key]["t2i"]["topK(1)"]}, {infos[key]["t2i"]["topK(5)"]}, {infos[key]["t2i"]["topK(10)"]}')
                        print_info += f' | {key} - i2t: {infos[key]["i2t"]["topK(1)"]}, {infos[key]["i2t"]["topK(5)"]}, {infos[key]["i2t"]["topK(10)"]}; t2i: {infos[key]["t2i"]["topK(1)"]}, {infos[key]["t2i"]["topK(5)"]}, {infos[key]["t2i"]["topK(10)"]}'
                    f.write('\n')
                print(print_info)
            torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    torch.distributed.barrier()

if __name__ == '__main__':

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    args = get_args_parser()
    args = args.parse_args()

    if args.cfg_options is not None:
        for key in args.cfg_options.keys():
            if 'loss' == key:
                args.__setattr__('loss', {item[0]: float(item[1]) for item in args.cfg_options['loss']})
            else:
                args.__setattr__(key, args.cfg_options[key])
        
        args.__delattr__('cfg_options')

    main(args)

    # upload to aliyunpan

    if args.aliyunpan_path is not None:
        process = subprocess.Popen(f'/workspace/aliyunpan/aliyunpan upload --norapid {args.output_dir} {args.aliyunpan_path}', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        
        while True:
            output = process.stdout.readline().decode('utf-8')
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        process.poll()
