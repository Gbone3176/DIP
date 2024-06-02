# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from model import DM2FNet_woPhy
from tools.config import OHAZE_ROOT
from datasets import OHazeDataset
from tools.utils import AvgMeter, check_mkdir, sliding_forward

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt/baseline_0512', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='O-Haze',
        help='experiment name.')
    args = parser.parse_args()

    return args

from torchvision import models
import torch.nn.functional as F

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

def calculate_perceptual_loss(output, target, vgg):
    output_features = vgg(output)
    target_features = vgg(target)
    perceptual_loss = 0
    for of, tf in zip(output_features, target_features):
        perceptual_loss += F.l1_loss(of, tf)
    return perceptual_loss



cfgs = {
    'use_physical': True,
    'iter_num': 40000,
    'train_batch_size': 32,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.95,
    'weight_decay': 2e-5,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 10000,
    'crop_size': 512,
}

def main():
    wandb.login(key='4b72fd67e7e6a8ebdc6f551acd7920ea62ecb664')
    wandb.init(project='hazy', entity='zimocc', config=cfgs, name='final_ohaze')

    net = DM2FNet_woPhy().cuda().train()
    vgg = VGG19(requires_grad=False).cuda()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad], 'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad], 'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, vgg, optimizer, log_path)


def train(net, vgg, optimizer, log_path):
    curr_iter = cfgs['last_iter']
    scaler = amp.GradScaler()
    torch.cuda.empty_cache()

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record = AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        perceptual_loss_record = AvgMeter()

        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) ** cfgs['lr_decay']

            haze, gt, _ = data

            batch_size = haze.size(0)

            haze, gt = haze.cuda(), gt.cuda()

            optimizer.zero_grad()

            with amp.autocast():
                x_jf, x_j1, x_j2, x_j3, x_j4 = net(haze)

                loss_x_jf = criterion(x_jf, gt)
                loss_x_j1 = criterion(x_j1, gt)
                loss_x_j2 = criterion(x_j2, gt)
                loss_x_j3 = criterion(x_j3, gt)
                loss_x_j4 = criterion(x_j4, gt)

                perceptual_loss = calculate_perceptual_loss(x_jf, gt, vgg)

                loss = loss_x_jf + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 + 0.1 * perceptual_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_record.update(loss.item(), batch_size)
            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)
            perceptual_loss_record.update(perceptual_loss.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_j1 %.5f], ' \
                  '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [perceptual loss %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg, perceptual_loss_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == 1 or (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, vgg, curr_iter, optimizer)
                torch.cuda.empty_cache()

            wandb.log({
                'iteration': curr_iter,
                'total_train_loss': train_loss_record.avg,
                'loss_x_fusion': loss_x_jf_record.avg,
                'loss_x_j1': loss_x_j1_record.avg,
                'loss_x_j2': loss_x_j2_record.avg,
                'loss_x_j3': loss_x_j3_record.avg,
                'loss_x_j4': loss_x_j4_record.avg,
                'perceptual_loss': perceptual_loss_record.avg,
                'learning_rate': optimizer.param_groups[1]['lr']
            })

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, vgg, curr_iter, optimizer):
    print('validating...')
    net.eval()
    criterion = nn.L1Loss().cuda()

    loss_record = AvgMeter()
    perceptual_loss_record = AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data
            haze, gt = haze.cuda(), gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            perceptual_loss = calculate_perceptual_loss(dehaze, gt, vgg)

            loss_record.update(loss.item(), haze.size(0))
            perceptual_loss_record.update(perceptual_loss.item(), haze.size(0))

    snapshot_name = 'final_ohaze_iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    log = '[validate]: [iter {}], [loss {:.5f}], [perceptual loss {:.5f}]'.format(curr_iter + 1, loss_record.avg, perceptual_loss_record.avg)
    print(log)
    open(log_path, 'a').write(log + '\n')
    torch.save(net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = OHazeDataset(OHAZE_ROOT, 'train_crop_512')
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4, shuffle=True, drop_last=True)

    val_dataset = OHazeDataset(OHAZE_ROOT, 'train_crop_512')
    val_loader = DataLoader(val_dataset, batch_size=1)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()
