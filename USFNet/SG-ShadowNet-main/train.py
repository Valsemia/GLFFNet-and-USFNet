from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch
from utils.utils import LambdaLR

from loss.losses import L_spa, preceptual_loss, edge_loss_with_mask
from data.datasets import ImageDataset, TestImageDataset

from networks import MyModel
from other_utils import init_net
import warnings


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=5, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--iter_loss', type=int, default=266, help='average loss for n iterations')
parser.add_argument('--input_nc', type=int, default=3, help='input channel')
parser.add_argument('--output_nc', type=int, default=3, help='output channel')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--log_path', type=str, default=r'ckpt_pre_Pc_BSB/log', help='# of gen filters in first conv layer')
opt = parser.parse_args()


# ISTD datasets
opt.dataroot = r'Dataset/ISTD_Dataset'

# checkpoint dir
if not os.path.exists('ckpt_pre_Pc_BSB'):
    os.mkdir('ckpt_pre_Pc_BSB')

print(opt)

net = MyModel(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.use_dropout)
net = init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[0])
net.cuda()

criterion_identity = torch.nn.L1Loss()
criterion_spa = L_spa()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(net.parameters()), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Dataset loader
dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True), batch_size=opt.batchSize, shuffle=True, num_workers=0)
test_dataloader = DataLoader(TestImageDataset(opt.dataroot), batch_size=1, shuffle=False, num_workers=0)

curr_iter = 0
G_losses_temp = 0
G_losses = []

# open(opt.log_path, 'w').write(str(opt) + '\n\n')
timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
log_path = os.path.join(opt.log_path, f'{timestamp}.txt')

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    net.train()
    for i, (s, sgt, mask) in enumerate(dataloader):
        # Set model input
        s = s.cuda()
        sgt = sgt.cuda()
        mask = mask.cuda()
        inv_mask = (1.0 - mask)  # non_shadow region mask
        s_without_mask = s * inv_mask
        shadow_only = s * mask

        # import torch
        # import matplotlib.pyplot as plt
        #
        # # 将 s_without_mask 从 GPU 迁移到 CPU，并转换为 NumPy 数组
        # s_without_mask_cpu = shadow_only.cpu().detach().numpy()
        #
        # # 通常情况下，图像张量是 (batch_size, channels, height, width)
        # # 我们需要移除 batch_size 维度并转置维度以便 Matplotlib 能正确显示
        # # 假设我们显示 batch 中的第一个图像
        # image_index = 0
        # image = s_without_mask_cpu[image_index]
        #
        # # 如果图像是三通道（例如 RGB），需要将通道维度移到最后
        # if image.shape[0] == 3:
        #     image = image.transpose(1, 2, 0)
        #
        # # 显示图像
        # plt.imshow(image)
        # plt.axis('off')  # 隐藏坐标轴
        # plt.show()
        
        ###### Generators ######
        optimizer_G.zero_grad()

        output = net(s, mask)
        loss_2 = criterion_identity(output, sgt) # 使用标识损失函数 criterion_identity 计算生成的结果图像 output 与原始去阴影图像 sgt 之间的损失 loss_2
        loss_spa = torch.mean(criterion_spa(output, sgt)) *10  # 使用空间一致性损失函数 criterion_spa 计算生成的结果图像 output 和原始去阴影图像 sgt 之间的空间一致性损失 loss_spa
        pre_loss = preceptual_loss(output, sgt)
        # loss_edge = edge_loss_with_mask(output, sgt, mask)
        # Total loss
        loss_G = loss_2 + loss_spa + pre_loss
        loss_G.backward()

        G_losses_temp += loss_G.item()

        optimizer_G.step()
        ###################################

        curr_iter += 1

        if (i+1) % 1 == 0:
            log = 'Epoch: %d, [iter 266\ %d], [loss_G %.5f], [loss_2 %.5f], [loss_shadow1 %.5f], [pre_loss %.5f]' % \
                  (epoch, i, loss_G, loss_2, loss_spa, pre_loss)
            print(log)

        if (i+1) % opt.iter_loss == 0:
            log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_2 %.5f], [loss_shadow1 %.5f], [pre_loss %.5f]' % \
                  (epoch, curr_iter, loss_G, loss_2, loss_spa, pre_loss)
            print(log)
            open(log_path, 'w').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)

            plt.figure(figsize=(10, 5))  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
            plt.plot(G_losses, 'b', label='loss')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
            # plt.plot(val_loss, 'r', label='val_loss')
            plt.ylabel('Loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')  # 个性化图例（颜色、形状等）
            plt.savefig(os.path.join(r'ckpt_pre_Pc_BSB', 'loss.png'))  # 保存图片 路径：/imgPath/
            plt.close('all')

            G_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f]'% (opt.iter_loss, G_losses[G_losses.__len__()-1])
            print(avg_log)
            open(log_path, 'w').write(avg_log + '\n')

    # Update learning rates
    lr_scheduler_G.step()

    if epoch >= (opt.n_epochs-50):
        torch.save(net.state_dict(), f'ckpt_pre_Pc_BSB/net_{epoch+1}.pth')

    print('Epoch:{}'.format(epoch))