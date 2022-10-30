import os
import math
import random
import argparse
from time import time
import glob
import sys
from pathlib import  Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

import timm
from timm.utils import accuracy
from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evalute(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_block=True)
        target = target.to(device, non_block=True)

        with torch.no_grad:
            output = model(images)
            loss = criterion(output, target)

        output = torch.nn.functional.softmax(output, dim=-1)
        acc1, acc5 = accuracy(output, target, topk=(1,5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.} ')

    return {k: meter.global_avg for k , meter in metric_logger.meters.items()}

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaer, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)

    print_freq = 2

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('lod_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(data_loader):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)

        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr

        loss = criterion(outputs, targets)
        loss /= accum_iter

        loss_scaer(loss, optimizer, clip_grad=max_norm,
                   parameters=model.parameters(), creat_graph=False,
                   update_grad=(data_iter_step + 1) % accum_iter == 0)

        loss_value = loss.item()

        if(data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', warmup_lr, epoch_1000x)
            print(f"Epoch: {epoch}, Step: {data_iter_step}, Loss:{loss}, Lr:{warmup_lr}")

def bulid_transform(is_train, args):
    if is_train:
        print("train train transform")
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(args.input_size, args.input_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])

    print("eval transform")
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(args.input_size, args.input_size),
            torchvision.transforms.ToTensor(),
        ])

def bulid_dataset(is_train, args):
    transform = bulid_transform(is_train, args)
    path = os.path.join(args.root_path, 'train' if is_train else 'test')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)

    return dataset

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre_training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,help='')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='一个batch更新一次梯度，如果内存或显存小，可以调整这个参数来增大batch')
    #Model parameters
    parser.add_argument('--model', default='mae_vit', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--mask_ratio',default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)
    #Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate；absolute_lr = base_lr * total_batch_size/256')
    parser.add_argument('--min_lr', type=float,default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args, mode='train', test_image_path=''):
    print(f"{mode} mode..")
    if mode == 'train':
        is_train = True
        distributed = False
        dataset_train = bulid_dataset(is_train=True, args=args)
        dataset_val = bulid_dataset(is_train=False, args=args)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        sampler_train = torch.utils.data.DataLoader(
            dataset_train, sample=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sample=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

        model = timm.create_model('resnet18',pretrained=True, num_classes=36,drop_rate=0.1, drop_path_rate=0.1)

        n_paramtters = sum(p.nume() for p in model.parameters() if p.requires_grad)
        print('number of params(M): %.2f' % (n_paramtters / 1.e6))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamw(model.parameters(), lr=args.lr, weight_decay=args.weight_decay )
        os.makedirs(args.log_dir, exist_ok=True)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
        model.eval()

        image = Image.open(test_image_path).convert('RGB')
        image = image.resize((args.input_size, args.input_size), Image.ANTIALIAS)
        image = torchvision.transforms.ToTensor(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)

        output = torch.nn.functional.softmax(output, dim=-1)
        class_idx = torch.argmax(output, dim=1)[0]
        score = torch.max(output, dim=1)[0][0]

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = 'train'











