from __future__ import print_function
import datetime
import logging
import os
import time
import sys
import numpy as np
import random
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import utils
from scheduler import WarmupMultiStepLR
from datasets.ntu60_sk import NTU60Subject_SK
import models.UST as Models
from ipdb import set_trace as st


def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training.log")), 
            logging.StreamHandler(sys.stdout) 
        ]
    )


def train_one_epoch(model, criterion,criterion_2, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    alpha = 0.05
    for (clip, target_sk), target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        clip, target,target_sk = clip.to(device), target.to(device),target_sk.to(device)

        idx = []
        for t in range(args.temporal_kernel_size//2 - 1, args.clip_len-args.temporal_kernel_size//2, args.temporal_stride):  
            idx.append(t)
        target_sk = target_sk[:,idx,:,:]

        output, output_sk= model(clip)
    
        loss_1 = alpha *criterion_2(output_sk, target_sk)
        loss_2 = criterion(output, target)
        
        optimizer.zero_grad()
        loss_1.backward(retain_graph=True)
        loss_2.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = clip.shape[0]
        metric_logger.update(loss=loss_2.item(),loss_sk=loss_1.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()

def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for (clip, target_sk), target, video_idx in metric_logger.log_every(data_loader, 100, header):
            clip = clip.to(device, non_blocking=True)
            target_sk = target_sk.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output,output_sk = model(clip)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            prob = F.softmax(input=output, dim=1)

            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    logging.info(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5))

    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v == label)
    class_acc = [c / float(s) if s > 0 else 0 for c, s in zip(class_correct, class_count)]

    logging.info(' * Video Acc@1 %f' % total_acc)
    logging.info(' * Class Acc@1 %s' % str(class_acc))

    return total_acc

def main(args):


    if args.output_dir:
        utils.mkdir(args.output_dir)
        setup_logging(args.output_dir)  

    logging.info(args)
    logging.info("torch version: %s", torch.__version__)
    logging.info("torchvision version: %s", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')

    logging.info("Loading data")
    dataset = NTU60Subject_SK(
        root=args.data_path,
        skeleton_root=args.sk_path,
        meta=args.data_meta,
        frames_per_clip=args.clip_len,
        step_between_clips=args.clip_step,
        num_points=args.num_points,
        train=True
    )

    dataset_test = NTU60Subject_SK(
        root=args.data_path,
        skeleton_root=args.sk_path,
        meta=args.data_meta,
        frames_per_clip=args.clip_len,
        step_between_clips=args.clip_step,
        num_points=args.num_points,
        train=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    logging.info("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  dim=args.dim, heads=args.heads,mlp_dim=args.mlp_dim, num_classes=dataset.num_classes,
                  dropout=args.dropout,depth = args.depth,hos_branches_num= args.hos_branches_num,encoder_channel =args.encoder_channel )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
                                     warmup_iters=warmup_iters, warmup_factor=1e-5)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    logging.info("Start training")
    start_time = time.time()
    acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, criterion_2,optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)
        acc = max(acc, evaluate(model, criterion, data_loader_test, device=device))

        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f'model_{epoch}.pth'))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Training time {total_time_str}')
    logging.info(f'Best Accuracy {acc}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='UST-SSM Model Training')

    parser.add_argument('--data-path', default='/data2/NTU120RGBD/pointcloud/ntu60npz2048', type=str, help='dataset')
    parser.add_argument('--sk-path', default='/data1/NTU120RGB/nturgb+d_skeletons_npy', type=str, help='dataset')
    parser.add_argument('--data-meta', default='/data2/NTU120RGBD/ntu60.list', help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='UST', type=str, help='model')
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--clip-step', default=2, type=int, metavar='N', help='steps between frame sampling')
    parser.add_argument('--num-points', default=2048, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('--radius', default=0.10, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    parser.add_argument('--dim', default=1024, type=int, help='ssm dim')  
    parser.add_argument('--depth', default=3, type=int, help='ssm depth') 
    parser.add_argument('--heads', default=8, type=int, help='ssm head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='mlp dim')
    parser.add_argument('--hos-branches-num', default=1, type=int)
    parser.add_argument('--encoder-channel', default=75, type=float)    

    parser.add_argument('--dropout', default=0.5, type=float, help='classifier dropout')
    parser.add_argument('-b', '--batch-size', default=12, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[10, 15], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/data2/POINT4D/UST-SSM/output/ntu_final2', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=1 python ntu-train.py