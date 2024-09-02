import os
import torch
import tqdm
import sys

import cv2
import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist

from torch.optim import Adam, SGD
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.dataloader import *
from lib.optim import *
from lib.IBandEDL_plus_pvt import IBandEDL_plus_pvt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch

# 设置随机种子
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True

def train(opt, args):

    # loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    loss_record = AvgMeter()
    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root, transform_list=opt.Train.Dataset.transform_list)

    if args.device_num > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=opt.Train.Dataloader.batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=opt.Train.Dataloader.num_workers,
                                    pin_memory=opt.Train.Dataloader.pin_memory,
                                    drop_last=True)

    total_step = len(train_loader)

    model = eval(opt.Model.name)()

    if args.device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            if 'backbone.layer' in name:
                backbone_params.append(param)
            else:
                pass
        else:
            decoder_params.append(param)

    # params_list = [{'params': backbone_params}, {
    #     'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]
    # optimizer = eval(opt.Train.Optimizer.type)(
    #     params_list, opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)
    
    optimizer = torch.optim.AdamW(model.parameters(), opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)
    
    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,
                                               minimum_lr=opt.Train.Scheduler.minimum_lr,
                                               max_iteration=len(
                                                   train_loader) * opt.Train.Scheduler.epoch,
                                               warmup_iteration=opt.Train.Scheduler.warmup_iteration)
    model.train()

    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(range(1, opt.Train.Scheduler.epoch + 1), desc='Epoch', total=opt.Train.Scheduler.epoch,
                               position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
    else:
        epoch_iter = range(1, opt.Train.Scheduler.epoch + 1)

    writer = SummaryWriter("./logs/pvt_urep")

    for epoch in epoch_iter:
        if args.local_rank <= 0 and args.verbose is True:
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
                train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
            if args.device_num > 1:
                train_sampler.set_epoch(epoch)
        else:
            step_iter = enumerate(train_loader, start=1)

        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True:
                with autocast():
                    sample = to_cuda(sample)
                    out = model(sample)

                    scaler.scale(out['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                sample = to_cuda(sample)
                out = model(sample, opt.Train.Dataloader.batch_size, opt.Train.Scheduler.epoch)

                # out = model(sample)
                out['loss'].backward()
                optimizer.step()
                scheduler.step()

            
            loss_record.update(out['loss'].data, opt.Train.Dataloader.batch_size)

            # loss_record2.update(out['loss2'].data, opt.Train.Dataloader.batch_size)
            # loss_record3.update(out['loss3'].data, opt.Train.Dataloader.batch_size)
            # loss_record4.update(out['loss4'].data, opt.Train.Dataloader.batch_size)
            # loss_record5.update(out['loss5'].data, opt.Train.Dataloader.batch_size)

            if args.local_rank <= 0 and args.verbose is True:
                step_iter.set_postfix({'loss': out['loss'].item()})

            if i % 20 ==0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[output: {:.4f}]'.
                      format(datetime.now(), epoch, opt.Train.Scheduler.epoch, i, total_step, loss_record.show()))
                # writer.add_scalar("loss_record2", loss_record2.show(), epoch)
                # writer.add_scalar("loss_record3", loss_record3.show(), epoch)
                # writer.add_scalar("loss_record4", loss_record4.show(), epoch)
                # writer.add_scalar("loss_record5", loss_record5.show(), epoch)
                writer.add_scalar("loss_record", out['loss'], epoch)

        if args.local_rank <= 0:
            os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(
                opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(
                ), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))

            if args.debug is True:
                debout = debug_tile(out)
                cv2.imwrite(os.path.join(
                    opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)


        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opt.Train.Checkpoint.checkpoint_dir,'pvt_urep-%d.pth' % epoch))
            print('[Saving Snapshot:]',  'pvt_urep-%d.pth' % epoch)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict()
        if args.device_num > 1
            else model.state_dict(), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


    writer.close()

if __name__ == '__main__':
    args = parse_args()
    opt = load_config(args.config)
    print("#"*20, "Start Training", "#"*20)
    train(opt, args)
