import os
import sys
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

# ====== your project imports ======
sys.path.append('./models')

from models.UncertaintyFusionSOD import UncertaintyFusionNet

from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from options import opt
# ==================================

# 可见 GPU
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")
cudnn.benchmark = True


# =========================
# 获取 local_rank
# =========================
def get_local_rank():
    # torchrun 风格
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    # launch 风格
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--local_rank="):
            return int(arg.split("=")[1])
        if arg == "--local_rank" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 0  # fallback 单卡


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return True, local_rank
    else:
        print("[WARN] Not using distributed mode")
        return False, 0


# =========================
# 损失函数
# =========================
class SaliencyBoundaryLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, tolerance=1,
                 pos_weight=None, reduction='mean', eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tolerance = tolerance
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def dice_loss_from_logits(self, logits, target):
        probs = torch.sigmoid(logits)
        return self.dice_loss_from_probs(probs, target)

    def dice_loss_from_probs(self, probs, target):
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (probs_flat * target_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps
        dice = (2.0 * intersection + self.eps) / denom
        return (1.0 - dice).mean()

    def tolerant_dice_loss(self, logits, target):
        probs = torch.sigmoid(logits)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)
        k = 2 * self.tolerance + 1
        pad = self.tolerance
        target_dil = F.max_pool2d(target, kernel_size=k, stride=1, padding=pad)
        return self.dice_loss_from_probs(probs, target_dil)

    def forward(self, b, gt_b):
        if gt_b.dtype not in (torch.float32, torch.float64):
            gt_b = gt_b.float()
        if b.dim() == 3:
            b = b.unsqueeze(1)
        if gt_b.dim() == 3:
            gt_b = gt_b.unsqueeze(1)
        bce = self.bce(b, gt_b)
        dice = self.dice_loss_from_logits(b, gt_b)
        tol_dice = self.tolerant_dice_loss(b, gt_b) if self.gamma != 0 else bce.new_tensor(0.0)
        return self.alpha * bce + self.beta * dice + self.gamma * tol_dice


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    return (1 - (inter + 1) / (union - inter + 1)).mean()


# =========================
# 训练与测试
# =========================
def train_one_epoch(train_loader, model, optimizer, epoch, save_path,
                    device, writer, is_main_process, total_step_ref):
    model.train()
    CE = nn.BCEWithLogitsLoss()
    boundary_loss = SaliencyBoundaryLoss()

    loss_all = 0.0
    epoch_step = 0

    for i, (images, gts, depth, boundarys) in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        boundarys = boundarys.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True).repeat(1, 3, 1, 1)

        s1, s2, s3, s4, kl_rgb, kl_depth, b1, b2, b3, b4 = model(images, depth)

        bce_iou = sum([CE(s, gts) + iou_loss(s, gts) for s in [s1, s2, s3, s4]])
        bce_boundary = sum([boundary_loss(b, boundarys) for b in [b1, b2, b3, b4]])

        loss = bce_iou + kl_rgb + kl_depth + bce_boundary
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        total_step_ref[0] += 1
        epoch_step += 1
        loss_all += loss.item()

        if is_main_process and (i % 100 == 0 or i == len(train_loader) or i == 1):
            mem_mb = torch.cuda.max_memory_allocated(device=device) / 1024.0 / 1024.0
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                  f'Step [{i:04d}/{len(train_loader):04d}], '
                  f'LR:{cur_lr:.7f}||sal_loss:{loss.item():.4f} mem:{mem_mb:.0f}MB')
            logging.info(f'#TRAIN#:Epoch [{epoch:03d}/{opt.epoch:03d}], '
                         f'Step [{i:04d}/{len(train_loader):04d}], '
                         f'LR:{cur_lr:.7f}, sal_loss:{loss.item():.4f}, mem_use:{mem_mb:.0f}MB')

            if writer:
                writer.add_scalar('Loss', loss.item(), global_step=total_step_ref[0])

    loss_all /= max(1, epoch_step)
    if is_main_process and writer:
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

    if is_main_process and (epoch % 5 == 0):
        torch.save(model.module.state_dict(), os.path.join(save_path, f'UncertaintyFusionNet_epoch_{epoch}.pth'))


def evaluate(test_loader, model, epoch, save_path, writer, is_main_process, device):
    if not is_main_process:
        return
    model.eval()
    with torch.no_grad():
        mae_sum = 0.0
        for _ in range(test_loader.size):
            image, gt, depth, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.to(device)
            depth = depth.to(device).repeat(1, 3, 1, 1)

            res, res2, res3, res4, *_ = model(image, depth)
            res = res + res2 + res3 + res4
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = torch.sigmoid(res).cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.abs(res - gt).mean()

        mae = mae_sum / test_loader.size
        if writer:
            writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print(f'Epoch: {epoch} MAE: {mae:.6f}')
        logging.info(f'#TEST#:Epoch:{epoch} MAE:{mae:.6f}')


def main():
    is_distributed, local_rank = init_distributed_mode()
    is_main_process = (not is_distributed) or (dist.get_rank() == 0)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    save_path = "Result_ALL_ddp_paper_234/"
    print("is_main_process:", is_main_process, " local_rank:", local_rank, " device:", device)
    # if is_main_process:
    #     os.makedirs(save_path, exist_ok=True)
    #     # logging.basicConfig(
    #     #     filename=os.path.join(save_path, 'UncertaintyFusionNet.log'),
    #     #     format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    #     #     level=logging.INFO, filemode='a',
    #     #     datefmt='%Y-%m-%d %I:%M:%S %p'
    #     # )
    if is_main_process:
        log_file = os.path.join(save_path, 'UncertaintyFusionNet.log')
        os.makedirs(save_path, exist_ok=True)
    else:
        log_file = os.devnull  # 其他进程不写日志

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )
    logging.info(f"Logger initialized on rank {local_rank}")



    image_root = "all"
    gt_root = opt.gt_root
    depth_root = opt.depth_root

    test_image_root = opt.test_rgb_root
    test_gt_root = opt.test_gt_root
    test_depth_root = opt.test_depth_root


    # test_image_root = '/home/b311/data2/Datasets/SOD/RGBTSOD/VT5000/Test/RGB/'
    # test_gt_root = '/home/b311/data2/Datasets/SOD/RGBTSOD/VT5000/Test/GT/'
    # test_depth_root = '/home/b311/data2/Datasets/SOD/RGBTSOD/VT5000/Test/T/'


    opt.batchsize = 6   #  max(1, int(opt.batchsize))

    model = UncertaintyFusionNet()
    if (opt.load_pre is not None) and is_main_process:
        model.load_pre(opt.load_pre)
        print('load model from ', opt.load_pre)
    # if is_main_process:
    #     model.load_state_dict(torch.load('./Result_ALL_ddp_paper/UncertaintyFusionNet_epoch_105.pth'), strict=True)
    #     print('load model from ', './Result_ALL_ddp_paper/UncertaintyFusionNet_epoch_105.pth')
    # #     # exit()
    

    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if is_main_process:
        num_parms = sum(p.numel() for p in model.parameters())
        print("Total Parameters (For Reference):", num_parms)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    temp_loader = get_loader(image_root, gt_root, depth_root,
                             batchsize=opt.batchsize, trainsize=opt.trainsize, boundary_flag=True)
    train_dataset = temp_loader.dataset if isinstance(temp_loader, DataLoader) else temp_loader

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.batchsize,
                                  sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opt.batchsize,
                                  shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize, boundary_flag=False)
    writer = SummaryWriter(os.path.join(save_path, 'summary')) if is_main_process else None

    total_step_ref = [0]
    for epoch in range(1, opt.epoch):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        if writer:
            writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        train_one_epoch(train_loader, model, optimizer, epoch, save_path,
                        device, writer, is_main_process, total_step_ref)

        # if epoch > 150 and is_main_process:
        if epoch > 100 and is_main_process:
            evaluate(test_loader, model, epoch, save_path, writer, is_main_process, device)

    if writer:
        writer.close()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
