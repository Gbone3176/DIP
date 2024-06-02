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

from model_hf import DM2FNet
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset
from tools.utils import AvgMeter, check_mkdir

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import wandb

import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 40000,
    'train_batch_size': 32,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.95,
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 5000,
    'crop_size': 256
}


# ------------------------
from torchvision import models
import torch.nn as nn


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


# ------------------------

def main():
    wandb.login(key='4b72fd67e7e6a8ebdc6f551acd7920ea62ecb664')
    wandb.init(project='hazy', entity='zimocc', config=cfgs, name='final_real2')

    net = DM2FNet().cuda().train()
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
    criterion = nn.L1Loss().cuda()

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4, shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=8)

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        perceptual_loss_record = AvgMeter()

        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) ** cfgs['lr_decay']

            haze, gt_trans_map, gt_ato, gt, _ = data
            batch_size = haze.size(0)

            haze = haze.cuda()
            gt_trans_map = gt_trans_map.cuda()
            gt_ato = gt_ato.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            x_jf, x_j0, x_j1, x_j2, x_j3, x_j4, t, a = net(haze)

            # 计算L1损失
            loss_x_jf = criterion(x_jf, gt)
            loss_x_j0 = criterion(x_j0, gt)
            loss_x_j1 = criterion(x_j1, gt)
            loss_x_j2 = criterion(x_j2, gt)
            loss_x_j3 = criterion(x_j3, gt)
            loss_x_j4 = criterion(x_j4, gt)
            loss_t = criterion(t, gt_trans_map)
            loss_a = criterion(a, gt_ato)

            # 计算感知损失
            perceptual_loss = calculate_perceptual_loss(x_jf, gt, vgg)

            alpha = min(1, float(curr_iter) / (cfgs['iter_num'] // 2))
            loss = (loss_x_jf + (loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4) 
                   + 10 * loss_t + loss_a + 0.1 * perceptual_loss)

            loss.backward()
            optimizer.step()

            # 更新损失记录器
            train_loss_record.update(loss.item(), batch_size)
            perceptual_loss_record.update(perceptual_loss.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [perceptual loss %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, perceptual_loss_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0 or curr_iter == 19500:
                validate(net, curr_iter, optimizer)

            wandb.log({
                'iteration': curr_iter,
                'total_train_loss': train_loss_record.avg,
                'perceptual_loss': perceptual_loss_record.avg,
                'learning_rate': optimizer.param_groups[1]['lr']
            })

            if curr_iter > cfgs['iter_num'] or curr_iter>19503:
                break



def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    # loss_record = AvgMeter()

    # with torch.no_grad():
    #     for data in tqdm(val_loader):
    #         haze, gt, _ = data

    #         haze = haze.cuda()
    #         gt = gt.cuda()

    #         dehaze = net(haze)

    #         loss = criterion(dehaze, gt)
    #         loss_record.update(loss.item(), haze.size(0))

    snapshot_name = 'final_iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, 0, optimizer.param_groups[1]['lr'])
    print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, 0))
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()







if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=8)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()

